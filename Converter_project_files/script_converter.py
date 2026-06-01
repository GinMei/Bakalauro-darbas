"""script_converter.py

Multi-stage Unity → Godot 4 C# conversion pipeline.

Architecture (5 stages per file):
    Stage 1 — CSharpStructuralParser   : split source into class/fields/methods
    Stage 2 — per-method LLM conversion: each method sent independently
    Stage 3 — ScriptReconstructor      : reassemble in correct order
    Stage 4 — hard validation          : Unity leakage + structure checks
    Stage 5 — failure handling         : stub written on any validation failure

No external API keys are required.  Ollama must be running at
http://localhost:11434 with a compatible model (default: codellama).
Override defaults with environment variables:

    OLLAMA_URL   — base URL     (default: http://localhost:11434)
    OLLAMA_MODEL — model name   (default: codellama)

Public API (unchanged from previous version):
    OllamaClient             — lightweight HTTP client for /api/generate
    ScriptConversionResult   — result for a single .cs file
    ScriptConverter          — converts one file or a whole batch
    write_fallback_stubs     — write placeholder stubs without Ollama
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger("script_converter")

# ---------------------------------------------------------------------------
# Model / server configuration
# ---------------------------------------------------------------------------

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")

# ---------------------------------------------------------------------------
# Stage 1 — C# Structural Parser
# ---------------------------------------------------------------------------

@dataclass
class ParsedMethod:
    """One method extracted from a Unity C# class."""
    name:      str   # method name (e.g. "Update", "MoveUp")
    full_text: str   # complete text: signature + body including braces


@dataclass
class ParsedScript:
    """Structural breakdown of a Unity C# source file."""
    raw_source:      str
    using_block:     str            # all `using ...;` lines
    namespace_open:  str            # "namespace Foo" or ""
    namespace_close: str            # "}" for namespace or ""
    attributes:      str            # [RequireComponent(...)] etc. before class
    class_header:    str            # e.g. "public class Foo : MonoBehaviour"
    class_name:      str            # e.g. "Foo"
    base_class:      str            # e.g. "MonoBehaviour"
    fields_block:    str            # content inside class before first method
    methods:         List[ParsedMethod] = dc_field(default_factory=list)
    trailing_block:  str = ""       # anything after last method before class close


class CSharpStructuralParser:
    """Parse a Unity C# source file into semantic blocks for per-method conversion.

    Uses brace-counting with comment/string awareness.  Does not require an
    external C# parser — handles the patterns common in Unity MonoBehaviour
    scripts reliably without being a full language parser.
    """

    _CLASS_RE = re.compile(
        r'(?:(?:public|internal|private|protected|abstract|sealed|static|partial)\s+)*'
        r'(?:class|struct)\s+(\w+)'
        r'(?:\s*<[^>]+>)?'
        r'(?:\s*:\s*(?:[\w<>\[\],\s.]+?))?'
        r'\s*\{',
        re.MULTILINE,
    )
    _NAMESPACE_RE = re.compile(r'^\s*namespace\s+([\w.]+)\s*\{', re.MULTILINE)

    # ------------------------------------------------------------------ public

    def parse(self, source: str) -> ParsedScript:
        using_block = self._extract_using_block(source)

        ns_match = self._NAMESPACE_RE.search(source)
        namespace_open  = f"namespace {ns_match.group(1)}" if ns_match else ""
        namespace_close = "}" if ns_match else ""

        cls = self._CLASS_RE.search(source)
        if not cls:
            raise ValueError("No class declaration found.")

        class_name  = cls.group(1)
        class_line  = cls.group(0)
        base_match  = re.search(r':\s*([\w<>\[\].,\s]+?)\s*(?:,|\{)', class_line)
        base_class  = base_match.group(1).strip() if base_match else ""
        class_header = class_line.rstrip("{").strip()

        # Attributes immediately before the class declaration
        pre_class = source[:cls.start()]
        attr_lines: List[str] = []
        for ln in reversed(pre_class.split("\n")):
            s = ln.strip()
            if s.startswith("["):
                attr_lines.insert(0, ln)
            elif s and not s.startswith("//"):
                break
        attributes = "\n".join(attr_lines)

        # Class body (content between outermost class braces)
        open_pos   = source.index("{", cls.start())
        class_body = self._extract_brace_content(source, open_pos)

        fields_block, methods, trailing_block = self._split_body(class_body)

        return ParsedScript(
            raw_source=source,
            using_block=using_block,
            namespace_open=namespace_open,
            namespace_close=namespace_close,
            attributes=attributes,
            class_header=class_header,
            class_name=class_name,
            base_class=base_class,
            fields_block=fields_block,
            methods=methods,
            trailing_block=trailing_block,
        )

    # ----------------------------------------------------------------- private

    @staticmethod
    def _extract_using_block(source: str) -> str:
        lines = []
        for ln in source.split("\n"):
            s = ln.strip()
            if s.startswith("using ") and s.endswith(";"):
                lines.append(ln)
            elif s and not s.startswith("//") and not s.startswith("/*"):
                break
        return "\n".join(lines)

    def _extract_brace_content(self, text: str, open_pos: int) -> str:
        """Return text between matching braces starting at open_pos."""
        depth = 0
        i = open_pos
        inner_start = open_pos + 1
        while i < len(text):
            c, i = self._next_char(text, i)
            if c is None:
                break
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[inner_start : i - 1]
        return text[inner_start:]

    def _split_body(self, body: str) -> Tuple[str, List[ParsedMethod], str]:
        """Split class body → (fields_block, methods, trailing_block)."""
        ranges = self._find_method_ranges(body)
        if not ranges:
            return body, [], ""
        first_start = ranges[0][0]
        last_end    = ranges[-1][1]
        fields_block   = body[:first_start].rstrip()
        trailing_block = body[last_end:].strip()
        methods = [
            ParsedMethod(name=name, full_text=body[s:e].strip())
            for s, e, name in ranges
        ]
        return fields_block, methods, trailing_block

    def _find_method_ranges(self, body: str) -> List[Tuple[int, int, str]]:
        """Return (start, end, name) for every method body in the class body.

        Strategy: at depth 0 relative to the class body, any ``{`` that is
        preceded by matching ``()`` (i.e. a parameter list) opens a method.
        Anything else (property, nested type, initialiser) is skipped.
        """
        results: List[Tuple[int, int, str]] = []
        n = len(body)
        i = 0
        seg_start  = 0      # start of current depth-0 segment
        has_parens = False  # have we seen matching () since seg_start?

        while i < n:
            c, new_i = self._next_char(body, i)
            if c is None:
                break
            i = new_i

            if c == "(":
                # Consume to matching ')'
                i = self._skip_matching(body, i - 1, "(", ")")
                has_parens = True

            elif c == "{":
                if has_parens:
                    # Opening brace of a method
                    sig_text    = body[seg_start : i - 1]
                    method_name = self._extract_method_name(sig_text)
                    end         = self._skip_matching(body, i - 1, "{", "}")
                    results.append((seg_start, end, method_name))
                    seg_start  = end
                    has_parens = False
                    i          = end
                else:
                    # Property / nested type / array initialiser — skip block
                    end        = self._skip_matching(body, i - 1, "{", "}")
                    seg_start  = end
                    has_parens = False
                    i          = end

            elif c == ";" and not has_parens:
                # Field declaration or abstract method — reset segment
                seg_start  = i
                has_parens = False

        return results

    # -------------------------------------------------------------- char utils

    @staticmethod
    def _next_char(text: str, pos: int) -> Tuple[Optional[str], int]:
        """Return (char, next_pos) skipping comments and string/char literals."""
        n = len(text)
        while pos < n:
            c = text[pos]
            # Single-line comment
            if c == "/" and pos + 1 < n and text[pos + 1] == "/":
                end = text.find("\n", pos)
                pos = (end + 1) if end != -1 else n
                continue
            # Block comment
            if c == "/" and pos + 1 < n and text[pos + 1] == "*":
                end = text.find("*/", pos + 2)
                pos = (end + 2) if end != -1 else n
                continue
            # Verbatim string @"..."
            if c == "@" and pos + 1 < n and text[pos + 1] == '"':
                pos += 2
                while pos < n:
                    if text[pos] == '"':
                        if pos + 1 < n and text[pos + 1] == '"':
                            pos += 2
                            continue
                        pos += 1
                        break
                    pos += 1
                continue
            # Regular string
            if c == '"':
                pos += 1
                while pos < n:
                    if text[pos] == "\\" :
                        pos += 2
                        continue
                    if text[pos] == '"':
                        pos += 1
                        break
                    pos += 1
                continue
            # Char literal
            if c == "'":
                pos += 1
                while pos < n:
                    if text[pos] == "\\":
                        pos += 2
                        continue
                    if text[pos] == "'":
                        pos += 1
                        break
                    pos += 1
                continue
            return c, pos + 1
        return None, pos

    def _skip_matching(self, text: str, open_pos: int, open_c: str, close_c: str) -> int:
        """Starting at open_pos (which IS open_c), return position AFTER matching close."""
        depth = 0
        i = open_pos
        n = len(text)
        while i < n:
            c, new_i = self._next_char(text, i)
            if c is None:
                break
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    return new_i
            i = new_i
        return n

    @staticmethod
    def _extract_method_name(sig: str) -> str:
        """Extract the method name from a signature string (text before '(')."""
        paren = sig.rfind("(")
        before = (sig[:paren] if paren != -1 else sig).strip()
        m = re.search(r"(\w+)\s*(?:<[^>]*>)?\s*$", before)
        return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Ollama HTTP client  (unchanged public API)
# ---------------------------------------------------------------------------

class OllamaClient:
    """Minimal HTTP client for the Ollama ``/api/generate`` endpoint."""

    def __init__(
        self,
        model:       str   = OLLAMA_MODEL,
        base_url:    str   = OLLAMA_URL,
        temperature: float = 0.1,
        max_tokens:  int   = 1024,
        timeout:     int   = 120,
    ) -> None:
        self.model       = model
        self.url         = base_url.rstrip("/") + "/api/generate"
        self._health_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    @property
    def health_url(self) -> str:
        return self._health_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 40960,
                },
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
        return response.json()["response"]

    def is_running(self) -> bool:
        try:
            r = requests.get(self._health_url, timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _safe_generate(client: OllamaClient, prompt: str, retries: int = 3) -> str:
    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(retries):
        try:
            return client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                delay = 2 ** attempt
                log.warning(
                    "[Ollama] attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, retries, exc, delay,
                )
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Stage 2 — Prompts
# ---------------------------------------------------------------------------

# Per-method prompt — sent once per method, much smaller than whole-file prompt
_PER_METHOD_PROMPT = """\
You are converting a single Unity C# method to Godot 4 C#.

OUTPUT RULES (MANDATORY):
- Output ONLY the converted method (signature + braced body)
- NO class declaration, NO using statements
- NO explanations, NO markdown, NO code fences
- PRESERVE all original author comments exactly as written
- DO NOT add any new comments
- Only add // TODO when an API is completely unmappable

CLASS CONTEXT (already converted — do NOT include in output):
{class_context}

ENGINE MAPPING (STRICT):

CLASS: MonoBehaviour → Node3D  |  ScriptableObject → Resource
LIFECYCLE: Awake/Start → _Ready()  |  Update → _Process(double delta)
           FixedUpdate → _PhysicsProcess(double delta)  |  LateUpdate → _Process(double delta)
           OnDestroy → _ExitTree()
           OnEnable/OnDisable/OnApplicationQuit → keep unchanged with // TODO
LOGGING: Debug.Log(x) → GD.Print(x)  |  Debug.LogWarning(x) → GD.PushWarning(x)
         Debug.LogError(x) → GD.PushError(x)
INPUT: Input.GetKey(KeyCode.X) → Input.IsKeyPressed(Key.X)
       Input.GetKeyDown(KeyCode.X) → Input.IsActionJustPressed("ui_accept")
       Input.GetKeyUp(KeyCode.X) → Input.IsActionJustReleased("ui_accept")
       Input.GetAxis("Horizontal") → Input.GetAxis("ui_right") - Input.GetAxis("ui_left")
       Input.mousePosition → GetViewport().GetMousePosition()
TIME: Time.deltaTime → (float)delta  |  Time.fixedDeltaTime → (float)delta
      Time.time → Time.GetTicksMsec() / 1000.0f  |  Time.timeScale → Engine.TimeScale
TRANSFORM: transform.position → Position  |  transform.eulerAngles → RotationDegrees
           transform.localScale → Scale  |  transform.parent → GetParent()
           transform.SetParent(t) → Reparent(t)
           transform.forward → -GlobalTransform.Basis.Z
           transform.Translate(v) → Translate(v)
           transform.LookAt(t) → LookAt(t, Vector3.Up)
VECTOR: Vector3.Distance → v1.DistanceTo(v2)  |  Vector3.Dot → v1.Dot(v2)
OBJECTS: GetComponent<T>() → keep unchanged with // TODO: map component to node manually
         GetComponentInChildren<T>() → keep unchanged with // TODO
         Instantiate(p) → p.Instantiate()  |  Destroy(obj) → obj.QueueFree()
         FindObjectOfType<T>() → GetTree().Root.FindChild("*", true, false) as T
PHYSICS: Rigidbody → RigidBody3D  |  rb.velocity → rb.LinearVelocity
         rb.AddForce(f) → rb.ApplyForce(f)  |  rb.AddImpulse(f) → rb.ApplyImpulse(f)
AUDIO: AudioSource → AudioStreamPlayer3D  |  audioSource.Play() → audioStreamPlayer.Play()
ANIMATION: Animator → AnimationPlayer  |  animator.Play("s") → animationPlayer.Play("s")
COROUTINES: DO NOT convert IEnumerator methods — keep IEnumerator return type unchanged
            DO NOT remove or replace yield return lines
            StartCoroutine/StopCoroutine → keep unchanged with // TODO
ATTRIBUTES: [SerializeField] → [Export]  |  [HideInInspector]/[Header]/[Tooltip] → remove
            [Range(a,b)] → remove  |  [ExecuteInEditMode] → [Tool]
CLASS DECLARATION: must be public partial class ClassName : Node3D
                   'partial' is REQUIRED — never omit it

FAILSAFE: if unsure, keep the original line unchanged — do NOT add comments.

<unity_method>
{method_code}
</unity_method>
"""

# Prompt for converting the fields block (attributes + field declarations)
_FIELDS_PROMPT = """\
Convert the following Unity C# field declarations to Godot 4 C#.

OUTPUT RULES:
- Output ONLY the converted field declarations (no class wrapper)
- NO explanations, NO markdown, NO code fences
- PRESERVE all original author comments exactly
- DO NOT add new comments

FIELD MAPPING:
- [SerializeField] → [Export]
- [HideInInspector] → remove entirely
- [Header("text")] → remove entirely
- [Tooltip("text")] → remove entirely
- [Range(a, b)] → remove entirely
- [RequireComponent(typeof(T))] → keep unchanged with // TODO: enforce dependency manually
- using UnityEngine; → remove (replaced by using Godot; at file level)

<unity_fields>
{fields_code}
</unity_fields>
"""


# ---------------------------------------------------------------------------
# Stage 3 — Deterministic class-header converter
# ---------------------------------------------------------------------------

_BASE_CLASS_MAP: Dict[str, str] = {
    "MonoBehaviour": "Node3D",
    "MonoBehavior":  "Node3D",  # common typo
    "ScriptableObject": "Resource",
}


def _convert_class_header(parsed: ParsedScript) -> str:
    """Convert the class declaration deterministically (no LLM)."""
    header = parsed.class_header
    for unity_base, godot_base in _BASE_CLASS_MAP.items():
        header = re.sub(
            rf'\b{re.escape(unity_base)}\b', godot_base, header
        )
    # Ensure 'partial' is present
    if "partial" not in header:
        header = re.sub(r'\bclass\b', "partial class", header, count=1)
    # Ensure 'public' is present
    if not header.strip().startswith("public"):
        header = "public " + header.lstrip()
    return header


def _convert_using_block(using_block: str) -> str:
    """Convert using statements: remove UnityEngine, ensure using Godot."""
    lines = []
    has_godot = False
    for ln in using_block.split("\n"):
        s = ln.strip()
        if "UnityEngine" in s:
            continue
        if "using Godot" in s:
            has_godot = True
        if s:
            lines.append(ln)
    if not has_godot:
        lines.insert(0, "using Godot;")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 3 — Reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(
    parsed:            ParsedScript,
    converted_header:  str,
    converted_usings:  str,
    converted_fields:  str,
    converted_methods: List[Tuple[str, bool]],  # (code, success)
) -> str:
    """Reassemble the full Godot C# source from converted parts."""
    parts: List[str] = []

    # Using block
    if converted_usings.strip():
        parts.append(converted_usings)
        parts.append("")

    # Namespace open
    if parsed.namespace_open:
        parts.append(parsed.namespace_open)
        parts.append("{")
        indent = "    "
    else:
        indent = ""

    # Attributes before class
    if parsed.attributes.strip():
        for ln in parsed.attributes.split("\n"):
            parts.append(indent + ln)

    # Class header
    parts.append(indent + converted_header)
    parts.append(indent + "{")

    # Fields
    if converted_fields.strip():
        for ln in converted_fields.split("\n"):
            parts.append(indent + "    " + ln if ln.strip() else ln)
        parts.append("")

    # Methods
    for code, _ in converted_methods:
        for ln in code.split("\n"):
            parts.append(indent + "    " + ln if ln.strip() else ln)
        parts.append("")

    # Trailing block (nested types, properties without method bodies, etc.)
    if parsed.trailing_block.strip():
        for ln in parsed.trailing_block.split("\n"):
            parts.append(indent + "    " + ln if ln.strip() else ln)
        parts.append("")

    parts.append(indent + "}")  # close class

    if parsed.namespace_close:
        parts.append("}")  # close namespace

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 4 — Hard validation
# ---------------------------------------------------------------------------

# Unity APIs that must NOT appear in Godot output
_UNITY_LEAKAGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\busing\s+UnityEngine\b'),
     "using UnityEngine namespace still present"),
    (re.compile(r'\bMonoBehaviour\b'),
     "MonoBehaviour base class not converted"),
    (re.compile(r'\bDebug\.Log\b'),
     "Debug.Log not converted (use GD.Print)"),
    (re.compile(r'\bTime\.deltaTime\b'),
     "Time.deltaTime not converted (use (float)delta)"),
]

# Critical structural requirements
_GODOT_REQUIRED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\busing\s+Godot\b'),
     "missing 'using Godot;'"),
    (re.compile(r'\bpublic\s+partial\s+class\b'),
     "class declaration missing 'public partial class'"),
    (re.compile(r'\bNode3D\b|\bNode\b|\bResource\b|\bControl\b|\bNode2D\b'),
     "no recognised Godot base class (Node3D / Node / Resource / Control)"),
]


def _validate_godot_csharp(text: str, parsed: Optional[ParsedScript] = None) -> str:
    """Return a non-empty error string if the output is invalid Godot C#.

    Checks (hard failures — all cause fallback stub):
      1. Unity API leakage (UnityEngine, MonoBehaviour, etc.)
      2. Godot structural requirements (using Godot, partial class, base class)
      3. File completeness (must end with '}')
      4. All original methods must be present in the output
    """
    stripped = text.strip()

    if not stripped:
        return "model returned empty output"

    # Unity leakage
    for pattern, label in _UNITY_LEAKAGE_PATTERNS:
        if pattern.search(stripped):
            return f"Unity API leakage: {label}"

    # Godot requirements
    for pattern, label in _GODOT_REQUIRED_PATTERNS:
        if not pattern.search(stripped):
            return f"Godot requirement not met: {label}"

    # Completeness
    if not stripped.endswith("}"):
        return "output may be truncated (does not end with '}')"

    # Method presence check
    if parsed is not None:
        for method in parsed.methods:
            if method.name not in ("unknown",) and method.name not in stripped:
                return f"method '{method.name}' missing from converted output"

    return ""


# ---------------------------------------------------------------------------
# Result type  (public API — unchanged)
# ---------------------------------------------------------------------------

@dataclass
class ScriptConversionResult:
    """Result of converting a single Unity C# file to Godot C#."""
    source_path: Path
    output_path: Path
    success:     bool
    csharp:      str = ""
    error:       str = ""
    warning:     str = ""


# ---------------------------------------------------------------------------
# ScriptConverter  (public API — same surface, redesigned internals)
# ---------------------------------------------------------------------------

class ScriptConverter:
    """Converts Unity C# scripts to Godot 4 C# via the 5-stage pipeline.

    Each file is:
      1. Structurally parsed into class/fields/methods
      2. Class header converted deterministically
      3. Each method converted independently via Ollama
      4. Reconstructed into a complete Godot C# file
      5. Hard-validated; fallback stub written on any failure
    """

    def __init__(
        self,
        model:       str   = OLLAMA_MODEL,
        base_url:    str   = OLLAMA_URL,
        batch_delay: float = 0.5,
    ) -> None:
        self._client      = OllamaClient(model=model, base_url=base_url)
        self._batch_delay = batch_delay
        self._parser      = CSharpStructuralParser()

    # -------------------------------------------------------------- public API

    def is_available(self) -> bool:
        available = self._client.is_running()
        if not available:
            log.debug(
                "Ollama not reachable at %s — conversion will use fallback stubs.",
                self._client.health_url,
            )
        return available

    def convert_file(
        self,
        cs_path:     Path,
        output_path: Path,
    ) -> ScriptConversionResult:
        """Convert one Unity .cs file through the full 5-stage pipeline."""
        result = ScriptConversionResult(
            source_path=cs_path,
            output_path=output_path,
            success=False,
        )

        # ── Read source ───────────────────────────────────────────────────────
        try:
            cs_source = cs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.error = f"Could not read {cs_path.name}: {exc}"
            return result

        if not cs_source.strip():
            result.error = f"{cs_path.name} is empty — skipped."
            return result

        # ── Stage 1: Structural parse ─────────────────────────────────────────
        try:
            parsed = self._parser.parse(cs_source)
        except ValueError as exc:
            log.warning("[Stage1] parse failed for %s: %s", cs_path.name, exc)
            _write_fallback(cs_path, cs_source, output_path, f"Parse failed: {exc}")
            result.error = str(exc)
            return result

        log.info(
            "[Stage1] parsed %s — %d method(s) found",
            cs_path.name, len(parsed.methods),
        )

        # ── Stage 2a: Deterministic header/using conversion ───────────────────
        converted_header = _convert_class_header(parsed)
        converted_usings = _convert_using_block(parsed.using_block)

        # ── Stage 2b: Fields via LLM ──────────────────────────────────────────
        converted_fields = self._convert_block_llm(
            parsed.fields_block,
            _FIELDS_PROMPT.format(fields_code=parsed.fields_block),
            cs_path.name,
            label="fields",
        ) if parsed.fields_block.strip() else parsed.fields_block

        # ── Stage 2c: Each method via LLM ────────────────────────────────────
        class_context = (
            f"{converted_header}\n"
            f"{{\n{converted_fields}\n}}"
        )
        converted_methods: List[Tuple[str, bool]] = []
        for method in parsed.methods:
            code, ok = self._convert_method_llm(method, class_context, cs_path.name)
            converted_methods.append((code, ok))

        # ── Stage 3: Reconstruct ──────────────────────────────────────────────
        reconstructed = _reconstruct(
            parsed,
            converted_header,
            converted_usings,
            converted_fields,
            converted_methods,
        )

        # ── Stage 4: Hard validation ──────────────────────────────────────────
        validation_error = _validate_godot_csharp(reconstructed, parsed)
        if validation_error:
            log.warning(
                "[Stage4] validation failed for %s: %s — writing fallback stub",
                cs_path.name, validation_error,
            )
            _write_fallback(cs_path, cs_source, output_path, validation_error)
            result.error = validation_error
            return result

        # ── Stage 5: Write output ─────────────────────────────────────────────
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(reconstructed, encoding="utf-8")
        except OSError as exc:
            result.error = f"Could not write {output_path}: {exc}"
            return result

        result.csharp   = reconstructed
        result.success  = True
        log.info("converted %s → %s", cs_path.name, output_path.name)
        return result

    def convert_batch(
        self,
        script_files: List[Path],
        project_root: Path,
        output_dir:   Path,
    ) -> List[ScriptConversionResult]:
        """Convert all .cs files, mirroring the Unity folder structure."""
        results: List[ScriptConversionResult] = []
        for i, cs_path in enumerate(script_files):
            output_path = _resolve_output_path(cs_path, project_root, output_dir)
            result = self.convert_file(cs_path, output_path)
            results.append(result)
            if i < len(script_files) - 1 and self._batch_delay > 0:
                time.sleep(self._batch_delay)
        ok   = sum(1 for r in results if r.success)
        fail = len(results) - ok
        log.info("batch complete  ok=%d  failed=%d", ok, fail)
        return results

    # ------------------------------------------------------------ private: LLM

    def _convert_method_llm(
        self,
        method:        ParsedMethod,
        class_context: str,
        filename:      str,
    ) -> Tuple[str, bool]:
        """Convert a single method via Ollama.  Returns (code, success)."""
        prompt = _PER_METHOD_PROMPT.format(
            class_context=class_context,
            method_code=method.full_text,
        )
        log.info("[Stage2] converting method '%s' in %s", method.name, filename)
        t0 = time.monotonic()

        raw = ""
        error_msg = ""

        try:
            raw = _safe_generate(self._client, prompt)
        except Exception as exc:
            error_msg = str(exc)
            log.error("[Ollama] model failed for %s::%s: %s", filename, method.name, exc)

        elapsed = time.monotonic() - t0

        if error_msg or not raw.strip():
            log.warning(
                "[Stage2] method '%s' conversion failed — keeping original",
                method.name,
            )
            return method.full_text, False

        # Strip fences if the model wrapped the output anyway
        if "```" in raw:
            raw = _strip_code_fences(raw)

        # Truncation detection per method
        if raw.strip() and not raw.strip().endswith("}"):
            log.warning(
                "[Stage2] method '%s' response truncated — keeping original",
                method.name,
            )
            return method.full_text, False

        log.info("[Stage2] method '%s' done (%.1fs)", method.name, elapsed)

        return raw.strip(), True

    def _convert_block_llm(
        self,
        original: str,
        prompt:   str,
        filename: str,
        label:    str,
    ) -> str:
        """Convert a non-method block (e.g. fields) via Ollama.
        Falls back to the original text on any failure."""
        if not original.strip():
            return original
        log.info("[Stage2] converting %s block in %s", label, filename)
        try:
            raw = _safe_generate(self._client, prompt)
            if "```" in raw:
                raw = _strip_code_fences(raw)
            return raw.strip() if raw.strip() else original
        except Exception as exc:
            log.warning("[Stage2] %s block conversion failed: %s — keeping original", label, exc)
            return original


# ---------------------------------------------------------------------------
# Module-level helpers  (public/private)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Extract C# source from within markdown code fences.

    Discards any prose before the opening fence and after the closing fence
    so that preamble text from chatty models does not contaminate the output.
    """
    text = text.strip()
    match = re.search(r'```(?:csharp|cs)?\s*\n(.*?)(?:\n```|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _write_fallback(
    cs_path:     Path,
    cs_source:   str,
    output_path: Path,
    error:       str,
) -> None:
    """Write a compilable placeholder stub and preserve the original source."""
    class_name = cs_path.stem
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stub = (
            "// CONVERSION FAILED — this is a placeholder stub.\n"
            f"// Original Unity C# source preserved as {cs_path.name}.original\n"
            f"// Error: {error}\n"
            "// Review and port manually.\n"
            "using Godot;\n"
            "\n"
            f"public partial class {class_name} : Node3D\n"
            "{\n"
            "}\n"
        )
        output_path.write_text(stub, encoding="utf-8")
        (output_path.parent / (cs_path.name + ".original")).write_text(
            cs_source, encoding="utf-8"
        )
    except OSError as exc:
        log.warning("could not write fallback for %s: %s", cs_path.name, exc)


def _resolve_output_path(
    cs_path:      Path,
    project_root: Path,
    output_dir:   Path,
) -> Path:
    """Compute destination .cs path, stripping the leading Assets/ component."""
    try:
        rel = cs_path.relative_to(project_root)
    except ValueError:
        return output_dir / "Scripts" / cs_path.name
    parts = rel.parts
    if parts and parts[0] == "Assets":
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(cs_path.name)
    return output_dir / rel


def write_fallback_stubs(
    script_files: List[Path],
    project_root: Path,
    output_dir:   Path,
) -> int:
    """Write compilable placeholder stubs for all .cs files without Ollama."""
    count = 0
    for cs_path in script_files:
        output_path = _resolve_output_path(cs_path, project_root, output_dir)
        try:
            cs_source = cs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            cs_source = ""
        _write_fallback(
            cs_path, cs_source, output_path,
            "Ollama unavailable — manual port required.",
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Preservation mode  (convert_scripts = False)
# ---------------------------------------------------------------------------

# Unity field types → closest Godot equivalents (deterministic, no AI)
_UNITY_TO_GODOT_FIELD_TYPES: Dict[str, str] = {
    "GameObject":           "Node3D",
    "Transform":            "Node3D",
    "Rigidbody":            "RigidBody3D",
    "Rigidbody2D":          "RigidBody2D",
    "Collider":             "CollisionShape3D",
    "BoxCollider":          "BoxShape3D",
    "SphereCollider":       "SphereShape3D",
    "CapsuleCollider":      "CapsuleShape3D",
    "Animator":             "AnimationPlayer",
    "AudioSource":          "AudioStreamPlayer3D",
    "Camera":               "Camera3D",
    "MeshRenderer":         "MeshInstance3D",
    "ParticleSystem":       "GpuParticles3D",
    "NavMeshAgent":         "NavigationAgent3D",
    "CharacterController":  "CharacterBody3D",
}

# Attributes that have no Godot equivalent — removed entirely
_PRESERVATION_REMOVE_ATTRS_INLINE = re.compile(
    r'\[\s*(?:HideInInspector|Header|Tooltip|Range|Space)(?:\s*\([^)]*\))?\s*\]\s*',
)
_PRESERVATION_ATTR_ONLY_LINE = re.compile(
    r'^\s*\[\s*(?:HideInInspector|Header|Tooltip|Range|Space)(?:\s*\([^)]*\))?\s*\]\s*$',
    re.MULTILINE,
)


def _preservation_fields_convert(fields_block: str) -> str:
    """Deterministic Unity → Godot field conversion for preservation mode.

    Strips incompatible attribute-only lines, then removes those same
    attributes when they appear inline on a field declaration line.
    Converts [SerializeField] to [Export] and maps known Unity field types
    to their Godot equivalents.  No logic is rewritten.
    """
    # Remove lines that are ONLY an incompatible attribute (e.g. [Header("x")])
    text = _PRESERVATION_ATTR_ONLY_LINE.sub('', fields_block)
    # Remove inline occurrences of those attributes on field declaration lines
    text = _PRESERVATION_REMOVE_ATTRS_INLINE.sub('', text)
    # [SerializeField] → [Export]
    text = re.sub(r'\[\s*SerializeField\s*\]', '[Export]', text)
    # Map known Unity field types to Godot equivalents
    for unity_type, godot_type in _UNITY_TO_GODOT_FIELD_TYPES.items():
        text = re.sub(rf'\b{re.escape(unity_type)}\b', godot_type, text)
    return text


def _build_preservation_stub(
    cs_path:     Path,
    cs_source:   str,
    output_path: Path,
) -> bool:
    """Build and write a Godot C# preservation stub from a Unity C# source.

    Preserved fields are converted deterministically; all method logic is
    wrapped in a block comment so it remains readable but does not compile.
    Returns True on success.
    """
    parser = CSharpStructuralParser()
    try:
        parsed = parser.parse(cs_source)
    except ValueError as exc:
        _write_fallback(cs_path, cs_source, output_path, f"Preservation: parse failed: {exc}")
        return False

    header = _convert_class_header(parsed)
    fields = _preservation_fields_convert(parsed.fields_block) if parsed.fields_block.strip() else ""

    commented_parts: List[str] = [m.full_text for m in parsed.methods]
    if parsed.trailing_block.strip():
        commented_parts.append(parsed.trailing_block)
    commented_block = "\n\n".join(commented_parts)

    ind = ""
    parts: List[str] = [
        "// Converted placeholder script (logic disabled)",
        "",
        "using Godot;",
        "",
    ]

    if parsed.namespace_open:
        parts.append(parsed.namespace_open)
        parts.append("{")
        ind = "    "

    for ln in parsed.attributes.split("\n"):
        if ln.strip():
            parts.append(ind + ln)

    parts.append(ind + header)
    parts.append(ind + "{")

    if fields.strip():
        parts.append(ind + "    // === Serialized fields (preserved) ===")
        for ln in fields.split("\n"):
            stripped = ln.strip()
            if stripped:
                parts.append(ind + "    " + stripped)
        parts.append("")

    if commented_block.strip():
        parts.append(ind + "    // === Original logic (commented out) ===")
        parts.append(ind + "    /*")
        for ln in commented_block.split("\n"):
            stripped = ln.strip()
            parts.append((ind + "    " + stripped) if stripped else "")
        parts.append(ind + "    */")

    parts.append(ind + "}")
    if parsed.namespace_close:
        parts.append("}")

    stub = "\n".join(parts) + "\n"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stub, encoding="utf-8")
        return True
    except OSError as exc:
        log.warning("could not write preservation stub for %s: %s", cs_path.name, exc)
        return False


def write_preservation_stubs(
    script_files: List[Path],
    project_root: Path,
    output_dir:   Path,
) -> Tuple[int, int]:
    """Write Godot C# preservation stubs for all Unity .cs files.

    Called when convert_scripts=False.  Each output file preserves all
    serialized fields (converted deterministically) and comments out all
    method logic, so scene/prefab script references remain valid without
    requiring AI conversion.

    Returns (success_count, failed_count).
    """
    success = 0
    failed  = 0
    for cs_path in script_files:
        output_path = _resolve_output_path(cs_path, project_root, output_dir)
        try:
            cs_source = cs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("could not read %s: %s", cs_path.name, exc)
            failed += 1
            continue
        if not cs_source.strip():
            failed += 1
            continue
        if _build_preservation_stub(cs_path, cs_source, output_path):
            success += 1
            log.info("preservation stub: %s → %s", cs_path.name, output_path.name)
        else:
            failed += 1
    return success, failed


# ---------------------------------------------------------------------------
# Godot C# → Unity C# deterministic converter  (no LLM)
# ---------------------------------------------------------------------------

@dataclass
class GodotScriptConversionResult:
    """Result of converting a single Godot C# file to Unity C#."""
    source_path: Path
    output_path: Path
    success:     bool
    csharp:      str = ""
    error:       str = ""


# Ordered replacement rules: (compiled_pattern, replacement)
# Each rule is applied in sequence to the full source text.
_G2U_RULES: List[Tuple[re.Pattern, str]] = [
    # using Godot → using UnityEngine
    (re.compile(r'\busing\s+Godot\s*;'),            "using UnityEngine;"),
    (re.compile(r'\busing\s+Godot\.Collections\s*;'), "using System.Collections.Generic;"),

    # Class declaration: public partial class X : Node3D → public class X : MonoBehaviour
    (re.compile(r'\bpublic\s+partial\s+class\s+(\w+)\s*:\s*Node3D\b'),
     r'public class \1 : MonoBehaviour'),
    (re.compile(r'\bpublic\s+partial\s+class\s+(\w+)\s*:\s*Node2D\b'),
     r'public class \1 : MonoBehaviour'),
    (re.compile(r'\bpublic\s+partial\s+class\s+(\w+)\s*:\s*Node\b'),
     r'public class \1 : MonoBehaviour'),
    (re.compile(r'\bpublic\s+partial\s+class\s+(\w+)\s*:\s*Control\b'),
     r'public class \1 : MonoBehaviour'),
    (re.compile(r'\bpublic\s+partial\s+class\s+(\w+)\s*:\s*Resource\b'),
     r'public class \1 : ScriptableObject'),

    # Lifecycle methods (override void _Ready → void Start, etc.)
    (re.compile(r'\boverride\s+void\s+_Ready\s*\(\s*\)'),
     'void Start()'),
    (re.compile(r'\boverride\s+void\s+_Process\s*\(\s*double\s+\w+\s*\)'),
     'void Update()'),
    (re.compile(r'\boverride\s+void\s+_PhysicsProcess\s*\(\s*double\s+\w+\s*\)'),
     'void FixedUpdate()'),
    (re.compile(r'\boverride\s+void\s+_ExitTree\s*\(\s*\)'),
     'void OnDestroy()'),
    (re.compile(r'\boverride\s+void\s+_EnterTree\s*\(\s*\)'),
     'void Awake()'),

    # delta usage inside methods  (float)delta → Time.deltaTime
    (re.compile(r'\(float\)\s*delta\b'),             'Time.deltaTime'),
    (re.compile(r'\(double\)\s*delta\b'),            'Time.deltaTime'),

    # Logging
    (re.compile(r'\bGD\.Print\s*\('),                'Debug.Log('),
    (re.compile(r'\bGD\.PushWarning\s*\('),          'Debug.LogWarning('),
    (re.compile(r'\bGD\.PushError\s*\('),            'Debug.LogError('),

    # Object lifecycle
    (re.compile(r'\bQueueFree\s*\(\s*\)'),           'Destroy(gameObject)'),

    # Attributes
    (re.compile(r'\[Export\]'),                      '[SerializeField]'),
    (re.compile(r'\[Tool\]'),                        '[ExecuteInEditMode]'),

    # Remove 'override' from Unity lifecycle methods (AI sometimes adds it)
    (re.compile(r'\boverride\s+((?:public|protected|private|internal)\s+)?void\s+(Start|Update|FixedUpdate|Awake|OnDestroy|LateUpdate)\s*\('),
     r'\1void \2('),
    # Remove 'override' when no access modifier (e.g. "override void Start(")
    (re.compile(r'\boverride\s+void\s+(Start|Update|FixedUpdate|Awake|OnDestroy|LateUpdate)\s*\('),
     r'void \1('),
]


# ---------------------------------------------------------------------------
# Gemini AI client  (hybrid pipeline for GodotToUnityCSharpConverter)
# ---------------------------------------------------------------------------

MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "gemini-3-flash-preview":        {"rpm": 5,  "rpd": 20},
    "gemini-3.1-flash-lite-preview": {"rpm": 15, "rpd": 500},
    "gemini-2.5-flash-lite": {"rpm": 10, "rpd": 20},
    "gemini-2.5-flash":      {"rpm": 5,  "rpd": 20},
}

# Ordered model chains per agent role — first available model wins
AGENT_MODEL_CHAINS: Dict[str, List[str]] = {
    "architecture":  ["gemini-3-flash-preview",         "gemini-2.5-flash"],
    "error_fixer":   ["gemini-2.5-flash",        "gemini-3.1-flash-lite-preview"],
}
_AGENT_TEMPERATURES: Dict[str, float] = {
    "architecture":  0.0,
    "error_fixer":   0.1,
    "type_mapper":   0.0,
    "api_converter": 0.0,
}

# Last-resort fallback when every chain model is exhausted
_EMERGENCY_FALLBACK = "gemini-3.1-flash-lite-preview"

# Derived flat list used only for _call_gemini_api error messages
_GEMINI_MODELS: List[str] = list(MODEL_LIMITS.keys())

_GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent"
)

# ---------------------------------------------------------------------------
# Deterministic type / API mappings  (applied before any AI call)
# ---------------------------------------------------------------------------

TYPE_MAP: Dict[str, str] = {
    "Node3D":                "MonoBehaviour",
    "Node2D":                "MonoBehaviour",
    "CharacterBody3D":       "MonoBehaviour",
    "Area3D":                "Collider",
    "MeshInstance3D":        "MeshRenderer",
    "RayCast3D":             "RaycastHit",
    "AnimationPlayer":       "Animator",
    "AnimationTree":         "Animator",
    "SceneTree":             "SceneManager",
    "Node":                  "GameObject",
    "RigidBody3D":           "Rigidbody",
    "CollisionShape3D":      "Collider",
    "CharacterController3D": "CharacterController",
    "ProgressBar":           "UnityEngine.UI.Slider",
}

API_MAP: Dict[str, str] = {
    "_Ready":          "Start",
    "_Process":        "Update",
    "_PhysicsProcess": "FixedUpdate",
    "_ExitTree":       "OnDestroy",
    "_EnterTree":      "Awake",
    "GD.Print":        "Debug.Log",
    "GD.PushWarning":  "Debug.LogWarning",
    "GD.PushError":    "Debug.LogError",
}

# Unity types that are invalid as user-defined class base classes, mapped to
# their correct replacements.  Applied deterministically after every AI pass.
_INVALID_BASE_CLASSES: Dict[str, str] = {
    "GameObject":          "MonoBehaviour",
    "Collider":            "MonoBehaviour",
    "Animator":            "MonoBehaviour",
    "MeshRenderer":        "MonoBehaviour",
    "Rigidbody":           "MonoBehaviour",
    "CharacterController": "MonoBehaviour",
    "SceneManager":        "MonoBehaviour",
}


def _fix_class_inheritance(code: str) -> str:
    """Replace illegal Unity base classes in class declarations and generic constraints.

    Runs deterministically so the AI cannot produce un-compilable inheritance.
    """
    for bad, good in _INVALID_BASE_CLASSES.items():
        # class Foo : BadBase  /  class Foo<T> : BadBase
        code = re.sub(
            r'(\bclass\s+\w+(?:\s*<[^>]+>)?\s*:\s*)' + re.escape(bad) + r'\b',
            r'\g<1>' + good,
            code,
        )
        # where T : BadBase  →  where T : Component
        code = re.sub(
            r'\bwhere\s+(\w+)\s*:\s*' + re.escape(bad) + r'\b',
            r'where \1 : Component',
            code,
        )
    return code


def _clean_placeholders(code: str) -> str:
    """Replace any leftover Godot placeholder markers with safe Unity equivalents."""
    code = code.replace("__RAYCAST_PLACEHOLDER__", "RaycastHit")
    code = code.replace("__SCENETREE_PLACEHOLDER__", "SceneManager")
    if "SceneManager" in code and "using UnityEngine.SceneManagement" not in code:
        lines = code.split("\n")
        last_using = -1
        for i, ln in enumerate(lines):
            if ln.strip().startswith("using "):
                last_using = i
        insert_at = last_using + 1 if last_using >= 0 else 0
        lines.insert(insert_at, "using UnityEngine.SceneManagement;")
        code = "\n".join(lines)
    return code


# Methods containing any of these strings are forwarded to the per-method AI pass
_COMPLEX_METHOD_TRIGGERS = [
    "RaycastHit",
    "SceneManager",
    "Animator",
    "connect(",
    "emit_signal(",
    "GetTree()",
    "GetNode(",
    "Godot.",
    "GpuParticles3D",
    "SubViewport",
    "PackedScene",
    "GD.Load",
    "AddChild(",
    "RemoveChild(",
    "BodyEntered",
    "BodyExited",
    "ChildEnteredTree",
    "ChildExitingTree",
    "QueueFree",
    "MoveAndSlide",
    "ReparentNode(",
    "GetChildren()",
    "GetChildCount()",
]

INVALID_PATTERNS = [
    "Node3D",
    "Node2D",
    "Area3D",
    "AnimationPlayer",
    "AnimationTree",
    "MeshInstance3D",
    "RayCast3D",
    "SceneTree",
    "RigidBody3D",
    "CollisionShape3D",
    "CharacterBody3D",
    "using Godot",
    "GD.Print",
    "GD.PushWarning",
    "GD.PushError",
    "GD.PrintErr",
    "GD.Load",
    "QueueFree()",
    "Godot.",
    "GpuParticles3D",
    "SubViewport",
    "PackedScene",
    "GetNode<",
    "GetNode(",
    "AddChild(",
    "RemoveChild(",
    "ChildEnteredTree",
    "ChildExitingTree",
    "BodyEntered",
    "BodyExited",
    "GetChildren()",
    "GetChildCount()",
    "MoveAndSlide()",
    "override void Start(",
    "override void Update(",
    "override void FixedUpdate(",
    "override void Awake(",
    "override void OnDestroy(",
]

# ---------------------------------------------------------------------------
# Stage-specific AI prompts
# ---------------------------------------------------------------------------


_ARCHITECTURE_PROMPT = """\
Convert this Godot-style C# script into proper Unity C#.

STRICT REQUIREMENTS — violating any of these makes the output invalid:
- Output ONLY the complete C# file — no markdown fences, no explanations, no prose
- The file must start with using statements (using UnityEngine; must be first or second)
- The class must inherit MonoBehaviour (or be static for utility classes)
- Do NOT use 'override' on Start, Update, FixedUpdate, Awake, or OnDestroy
- Do NOT include 'using' statements inside the class body — only at the top of the file
- Remove ALL Godot concepts completely — rewrite logic using Unity architecture:
  - GetNode<T>() → serialized [SerializeField] fields or GetComponent<T>()
  - AddChild() / RemoveChild() → Instantiate() / Destroy()
  - GpuParticles3D → ParticleSystem
  - SubViewport → RenderTexture or remove
  - PackedScene → GameObject prefab loaded via Resources.Load<GameObject>()
  - BodyEntered / BodyExited signals → OnTriggerEnter / OnTriggerExit
  - ChildEnteredTree / ChildExitingTree → OnTriggerEnter / OnTriggerExit
  - QueueFree() → Destroy(gameObject)
  - MoveAndSlide() → CharacterController.Move()
  - GetTree().ChangeScene() → SceneManager.LoadScene()
  - RaycastHit field (from RayCast3D) → Physics.Raycast with RaycastHit out parameter
  - SceneManager → add using UnityEngine.SceneManagement; at top of file
- Output MUST compile in Unity without any errors

CODE:
{code}
"""

_ERROR_FIX_PROMPT = """\
Fix this Unity C# script. It has the following errors:

{errors}

STRICT:
- Output ONLY the complete fixed C# file — no markdown fences, no explanations
- Remove ALL Godot types and replace with Unity equivalents
- Fix ALL CS0246 and other type-not-found errors
- Do NOT use 'override' on Start, Update, FixedUpdate, Awake, or OnDestroy
- Do NOT place 'using' statements inside the class body
- Remove all remaining Godot APIs: GetNode, AddChild, QueueFree, GpuParticles3D, SubViewport, PackedScene
- Ensure using UnityEngine; is present at top of file
- Preserve all logic
- Output MUST compile in Unity without errors

CODE:
{code}
"""


def _call_gemini_api(model: str, prompt: str, api_key: str, temperature: float = 0.1) -> str:
    """Call the Gemini generateContent REST endpoint and return generated text."""
    import json as _json
    url = _GEMINI_API_URL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        data=_json.dumps(payload),
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response structure: {exc}") from exc


class ModelUsageTracker:
    """Tracks per-model RPM and RPD usage to prevent quota violations."""

    def __init__(self) -> None:
        self._calls_today:  Dict[str, int]         = defaultdict(int)
        self._calls_minute: Dict[str, List[float]]  = defaultdict(list)
        self._failed_models: set                    = set()

    def mark_failed(self, model: str) -> None:
        """Permanently blacklist *model* for the rest of this conversion run."""
        if model not in self._failed_models:
            self._failed_models.add(model)
            log.warning("[Tracker] %s blacklisted for this run (hard failure)", model)

    def can_use(self, model: str) -> bool:
        if model in self._failed_models:
            return False
        limits = MODEL_LIMITS.get(model)
        if not limits:
            return True
        if self._calls_today[model] >= limits["rpd"]:
            log.debug("[Tracker] %s daily limit reached (%d/%d)", model,
                      self._calls_today[model], limits["rpd"])
            return False
        now = time.time()
        self._calls_minute[model] = [
            t for t in self._calls_minute[model] if now - t < 60
        ]
        if len(self._calls_minute[model]) >= limits["rpm"]:
            log.debug("[Tracker] %s RPM limit reached (%d/%d)", model,
                      len(self._calls_minute[model]), limits["rpm"])
            return False
        return True

    def record(self, model: str) -> None:
        self._calls_today[model] += 1
        self._calls_minute[model].append(time.time())

    def status(self) -> Dict[str, Dict[str, int]]:
        now = time.time()
        return {
            m: {
                "today": self._calls_today[m],
                "last_minute": sum(1 for t in self._calls_minute[m] if now - t < 60),
            }
            for m in _GEMINI_MODELS
        }


class GeminiClient:
    """Rate-aware Gemini client with per-agent model chains and RPM/RPD tracking.

    Call ``generate_with_agent(agent_type, prompt)`` using one of:
        type_mapper   — lite models (bulk, simple type fixes)
        api_converter — lite models (API name substitution)
        architecture  — strong models (full script restructure)
        error_fixer   — strong models (iterative compile-error repair)
    """

    # Class-level tracker so quota is shared across all instances in a session
    _tracker: ModelUsageTracker = ModelUsageTracker()

    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "")

    # ---------------------------------------------------------------- public

    def generate(self, prompt: str) -> Optional[str]:
        """Convenience wrapper — routes to the 'architecture' agent chain."""
        return self.generate_with_agent("architecture", prompt)

    def generate_with_role(self, prompt: str, role: str = "balanced") -> Optional[str]:
        """Backward-compatible shim: maps old role names to agent types."""
        _role_to_agent = {
            "fast":     "type_mapper",
            "balanced": "architecture",
            "fallback": "error_fixer",
        }
        return self.generate_with_agent(_role_to_agent.get(role, "architecture"), prompt)

    def generate_with_agent(self, agent_type: str, prompt: str) -> Optional[str]:
        """Generate using the model chain assigned to *agent_type*.

        Iterates the chain in order; skips models that have hit RPM/RPD limits
        (with a brief sleep on RPM exhaustion); falls back to the emergency
        fallback model if the entire chain is unavailable.
        """
        if not self.api_key:
            log.debug("[Gemini] GEMINI_API_KEY not set — skipping")
            return None

        chain = AGENT_MODEL_CHAINS.get(agent_type, AGENT_MODEL_CHAINS["architecture"])
        temperature = _AGENT_TEMPERATURES.get(agent_type, 0.0)

        for model in chain:
            if not self._tracker.can_use(model):
                time.sleep(2)
                if not self._tracker.can_use(model):
                    log.info("[Gemini] %s still rate-limited — skipping", model)
                    continue
            try:
                result = _call_gemini_api(model, prompt, self.api_key, temperature)
                if result and result.strip():
                    self._tracker.record(model)
                    log.info("[Gemini] %s responded (agent=%s)", model, agent_type)
                    return result.strip()
            except Exception as exc:
                log.warning("[Gemini] %s failed: %s — blacklisting for this run", model, exc)
                self._tracker.mark_failed(model)
                continue

        # Emergency fallback — use the cheapest high-quota model
        if self._tracker.can_use(_EMERGENCY_FALLBACK):
            try:
                result = _call_gemini_api(_EMERGENCY_FALLBACK, prompt, self.api_key, temperature)
                if result and result.strip():
                    self._tracker.record(_EMERGENCY_FALLBACK)
                    log.warning(
                        "[Gemini] used emergency fallback %s for agent=%s",
                        _EMERGENCY_FALLBACK, agent_type,
                    )
                    return result.strip()
            except Exception as exc:
                log.error("[Gemini] emergency fallback failed: %s — blacklisting", exc)
                self._tracker.mark_failed(_EMERGENCY_FALLBACK)

        log.error("[Gemini] all models exhausted for agent=%s", agent_type)
        return None


# ---------------------------------------------------------------------------
# Godot C# → Unity C# 6-stage hybrid converter
# ---------------------------------------------------------------------------

class GodotToUnityCSharpConverter:
    """6-stage hybrid Godot 4 C# → Unity C# converter.

    Stage 1  Preprocessor        — structural parse (CSharpStructuralParser), no AI
    Stage 2  Type mapping        — deterministic TYPE_MAP; AI fast-model if unknowns remain
    Stage 3  API conversion      — deterministic API_MAP + _G2U_RULES; AI per complex method
    Stage 4  Architecture agent  — AI balanced-model rewrites full script
    Stage 5  Compilation validator — code-only pattern check (no AI)
    Stage 6  Error-fix loop      — AI fallback-model, up to MAX_ITERATIONS=1
    """

    MAX_ITERATIONS = 1

    def __init__(self) -> None:
        self.gemini          = GeminiClient()
        self._parser         = CSharpStructuralParser()
        self._ollama_fallback = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    # ---------------------------------------------------------------- public API

    def convert_file(
        self,
        cs_path:     Path,
        output_path: Path,
        errors:      str = "",
    ) -> GodotScriptConversionResult:
        """Convert one Godot .cs file through the 6-stage pipeline.

        ``errors`` may carry Unity compiler error strings (CS0246 etc.) from a
        prior build attempt; they seed Stage 6 on the first iteration.
        """
        result = GodotScriptConversionResult(
            source_path=cs_path,
            output_path=output_path,
            success=False,
        )

        try:
            source = cs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.error = f"Cannot read {cs_path.name}: {exc}"
            return result

        if not source.strip():
            result.error = f"{cs_path.name} is empty — skipped."
            return result

        code = self._run_pipeline(source, cs_path.name, errors)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            result.error = f"Cannot write {output_path}: {exc}"
            return result

        validation_errors = self._validate(code)
        if validation_errors:
            log.warning(
                "[G2U] residual issues in %s: %s",
                cs_path.name, "; ".join(validation_errors),
            )
            result.error = "; ".join(validation_errors)

        result.csharp  = code
        result.success = True
        log.info("g2u converted %s → %s", cs_path.name, output_path.name)
        return result

    # -------------------------------------------------------------- pipeline

    def _run_pipeline(self, source: str, filename: str, initial_errors: str = "") -> str:
        # Stage 1: Preprocessor — structural parse, no AI
        code = self._preprocess(source, filename)

        # Stage 2: Type mapping — deterministic
        code = self._apply_type_mapping(code)

        # Stage 3: API conversion — deterministic rules
        code = self._apply_api_mapping(code)
        code = self._apply_rules(code)
        code = _fix_class_inheritance(code)
        code = _clean_placeholders(code)

        # Stage 4: Architecture agent — strong models (gemini-3-flash-preview first)
        log.info("[G2U][S4] architecture agent — %s", filename)
        arch = self.gemini.generate_with_agent(
            "architecture", _ARCHITECTURE_PROMPT.format(code=code)
        )
        if arch:
            arch_clean = _strip_code_fences(arch) if "```" in arch else arch
            if self._is_valid_arch_output(arch_clean):
                code = arch_clean
            else:
                log.warning(
                    "[G2U][S4] architecture output failed structural check for %s — keeping previous",
                    filename,
                )

        # Post-architecture deterministic cleanup — re-apply in case AI re-introduced issues
        code = _fix_class_inheritance(code)
        code = _clean_placeholders(code)

        # Stage 5+6: Validate then iterative error-fix loop
        seed_errors: List[str] = initial_errors.splitlines() if initial_errors else []
        for iteration in range(self.MAX_ITERATIONS):
            validation_errors = self._validate(code)
            all_errors = validation_errors + seed_errors
            if not all_errors:
                log.info("[G2U][S6] %s clean after %d iteration(s)", filename, iteration)
                break
            log.info(
                "[G2U][S6] iteration %d/%d — %d error(s) in %s",
                iteration + 1, self.MAX_ITERATIONS, len(all_errors), filename,
            )
            fixed = self.gemini.generate_with_agent(
                "error_fixer",
                _ERROR_FIX_PROMPT.format(errors="\n".join(all_errors), code=code),
            )
            if fixed:
                code = _strip_code_fences(fixed) if "```" in fixed else fixed
                code = _fix_class_inheritance(code)
                code = _clean_placeholders(code)
            seed_errors = []  # compiler errors only seeded on first iteration
        else:
            remaining = self._validate(code)
            if remaining:
                log.error(
                    "[G2U][S6] %s — %d error(s) unresolved after %d iteration(s):\n%s",
                    filename,
                    len(remaining),
                    self.MAX_ITERATIONS,
                    "\n".join(f"  {e}" for e in remaining),
                )

        return code

    # ----------------------------------------------------------- stage helpers

    def _preprocess(self, source: str, filename: str) -> str:
        """Stage 1: validate structure with CSharpStructuralParser; return source unchanged."""
        try:
            self._parser.parse(source)
            log.info("[G2U][S1] structure OK — %s", filename)
        except ValueError as exc:
            log.warning("[G2U][S1] parse warning for %s: %s", filename, exc)
        return source

    @staticmethod
    def _apply_type_mapping(code: str) -> str:
        """Stage 2 deterministic: replace Godot type names via TYPE_MAP."""
        for godot_type, unity_type in TYPE_MAP.items():
            code = re.sub(rf'\b{re.escape(godot_type)}\b', unity_type, code)
        return code

    @staticmethod
    def _contains_unknown_types(code: str) -> bool:
        """Return True if bare Godot.* type references survive after TYPE_MAP."""
        return "Godot." in code

    @staticmethod
    def _apply_api_mapping(code: str) -> str:
        """Stage 3 deterministic: replace Godot API names via API_MAP."""
        for godot_api, unity_api in API_MAP.items():
            code = re.sub(rf'\b{re.escape(godot_api)}\b', unity_api, code)
        return code

    @staticmethod
    def _is_valid_arch_output(text: str) -> bool:
        """Return False if architecture agent output has structural red flags."""
        stripped = text.strip()
        # Must have a class declaration
        if not re.search(r'\bclass\s+\w+', stripped):
            return False
        # Must end with a closing brace
        if not stripped.endswith("}"):
            return False
        # Must contain UnityEngine reference
        if "UnityEngine" not in stripped and "MonoBehaviour" not in stripped:
            return False
        # Detect using-inside-class: a 'using' statement that appears after the first '{'
        first_brace = stripped.find("{")
        if first_brace != -1:
            after_class_open = stripped[first_brace:]
            # Only top-level using statements are valid (before any '{')
            for line in after_class_open.split("\n")[1:]:
                if re.match(r'\s*using\s+\w+', line):
                    return False
        return True

    @staticmethod
    def _needs_fix(code: str) -> bool:
        return any(p in code for p in INVALID_PATTERNS)

    @staticmethod
    def _validate(text: str) -> List[str]:
        """Stage 5: return list of error strings for any remaining invalid patterns."""
        errors: List[str] = []
        for pattern in INVALID_PATTERNS:
            if pattern in text:
                errors.append(f"Invalid type: {pattern}")
        if "using UnityEngine" not in text:
            errors.append("Missing UnityEngine namespace")
        return errors

    @staticmethod
    def _apply_rules(source: str) -> str:
        """Apply existing _G2U_RULES (lifecycle, attributes, delta, QueueFree, etc.)."""
        text = source
        for pattern, replacement in _G2U_RULES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _build_error_prompt(code: str, errors: str) -> str:
        return _ERROR_FIX_PROMPT.format(errors=errors, code=code)


# ---------------------------------------------------------------------------
# GDScript → Unity C# converter
# ---------------------------------------------------------------------------

def _gd_class_name(stem: str) -> str:
    """Convert a snake_case GDScript filename stem to PascalCase class name."""
    return "".join(w.capitalize() for w in stem.split("_")) or stem


_GD_TO_UNITY_PROMPT = """\
Convert this GDScript file to Unity C#.

OUTPUT RULES (MANDATORY):
- Output ONLY the complete C# file — no markdown fences, no explanations, no prose
- File must start with using statements; using UnityEngine; must be present
- Class name must be: {class_name}
- Class must inherit MonoBehaviour (unless it is a pure data class)
- Do NOT use 'override' on Start, Update, FixedUpdate, Awake, or OnDestroy
- Output MUST compile in Unity without errors

GDSCRIPT → UNITY MAPPING

CLASS:
  extends Node3D / Node / Node2D / CharacterBody3D  →  public class {class_name} : MonoBehaviour
  extends Resource                                  →  public class {class_name} : ScriptableObject
  (no extends)                                      →  public class {class_name} : MonoBehaviour

LIFECYCLE:
  func _ready()                    →  void Start()
  func _process(delta)             →  void Update()  (replace delta with Time.deltaTime)
  func _physics_process(delta)     →  void FixedUpdate()
  func _exit_tree()                →  void OnDestroy()
  func _enter_tree()               →  void Awake()
  func _unhandled_input(event)     →  // TODO: handle input in Update()
  func _on_X_signal(...)           →  void OnXSignal(...) matching UnityEvent pattern

VARIABLES:
  @export var x: float = 1.0  →  [SerializeField] private float x = 1.0f;
  var x: int = 0              →  private int x = 0;
  @onready var x = $Node      →  [SerializeField] private GameObject x;  // assign in Inspector
  const X = 5                 →  private const int X = 5;
  static var x                →  private static <type> x;

TYPES:
  float → float  |  int → int  |  bool → bool  |  String → string
  Vector3 → Vector3  |  Vector2 → Vector2  |  Quaternion → Quaternion
  Color → Color  |  Array → List<object>  |  Dictionary → Dictionary<string, object>
  NodePath → string  |  Variant → object  |  Node → GameObject

NODE REFERENCES:
  $NodeName / get_node("NodeName")  →  [SerializeField] private GameObject nodeNameRef; // assign in Inspector
  get_parent()                      →  transform.parent.gameObject
  get_children()                    →  GetComponentsInChildren<Transform>(true)
  find_child("name")                →  transform.Find("name")?.gameObject
  add_child(node)                   →  // TODO: use Instantiate() and node.transform.SetParent(transform)
  remove_child(node)                →  Destroy(node)
  is_inside_tree()                  →  gameObject.activeInHierarchy

SIGNALS:
  signal name                       →  public event System.Action name;
  signal name(arg: Type)            →  public event System.Action<Type> name;
  emit_signal("name")               →  name?.Invoke();
  emit_signal("name", arg)          →  name?.Invoke(arg);
  connect("name", target, "method") →  name += target.method;  // TODO: verify signature

LOGGING:
  print(x)        →  Debug.Log(x)
  push_warning(x) →  Debug.LogWarning(x)
  push_error(x)   →  Debug.LogError(x)
  printerr(x)     →  Debug.LogError(x)

OBJECT LIFECYCLE:
  queue_free()                     →  Destroy(gameObject)
  PackedScene.instantiate()        →  Instantiate(prefab)
  duplicate()                      →  Instantiate(gameObject)

INPUT:
  Input.is_key_pressed(KEY_X)          →  Input.GetKey(KeyCode.X)
  Input.is_action_pressed("name")      →  Input.GetButton("name")
  Input.is_action_just_pressed("name") →  Input.GetButtonDown("name")
  Input.is_action_just_released("name") →  Input.GetButtonUp("name")
  Input.get_axis("neg", "pos")         →  Input.GetAxis("Horizontal")
  Input.get_vector(...)                →  new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"))

PHYSICS:
  move_and_slide()        →  GetComponent<CharacterController>().Move(velocity * Time.deltaTime)
  apply_central_force(f)  →  GetComponent<Rigidbody>().AddForce(f)
  apply_central_impulse(f) →  GetComponent<Rigidbody>().AddForce(f, ForceMode.Impulse)
  linear_velocity         →  GetComponent<Rigidbody>().velocity

SCENE / APP:
  get_tree().change_scene_to_file(path) →  UnityEngine.SceneManagement.SceneManager.LoadScene(path)
  get_tree().quit()                     →  Application.Quit()
  get_tree().paused                     →  Time.timeScale == 0

MATH:
  randf()           →  Random.value
  randi()           →  Random.Range(0, int.MaxValue)
  randf_range(a,b)  →  Random.Range(a, b)
  clamp(v,lo,hi)    →  Mathf.Clamp(v, lo, hi)
  lerp(a,b,t)       →  Mathf.Lerp(a, b, t)
  abs(x)            →  Mathf.Abs(x)
  PI                →  Mathf.PI

GDSCRIPT SOURCE:
{gd_source}
"""


@dataclass
class GDScriptConversionResult:
    """Result of converting a single GDScript file to Unity C#."""
    source_path: Path
    output_path: Path
    success:     bool
    csharp:      str = ""
    error:       str = ""


def _write_gd_fallback(
    gd_path:     Path,
    gd_source:   str,
    output_path: Path,
    error:       str,
) -> None:
    """Write a compilable placeholder stub and preserve the original GDScript source."""
    class_name = _gd_class_name(gd_path.stem)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stub = (
            "// GDSCRIPT CONVERSION FAILED — this is a placeholder stub.\n"
            f"// Original GDScript preserved as {gd_path.name}.original\n"
            f"// Error: {error}\n"
            "// Review and port manually.\n"
            "using UnityEngine;\n"
            "\n"
            f"public class {class_name} : MonoBehaviour\n"
            "{\n"
            "}\n"
        )
        output_path.write_text(stub, encoding="utf-8")
        (output_path.parent / (gd_path.name + ".original")).write_text(
            gd_source, encoding="utf-8"
        )
    except OSError as exc:
        log.warning("could not write GDScript fallback for %s: %s", gd_path.name, exc)


class GDScriptToUnityCSharpConverter:
    """Convert GDScript (.gd) files to Unity C# MonoBehaviours.

    Uses Gemini as the primary LLM (with the 'architecture' agent chain) and
    falls back to Ollama if no Gemini API key is set.  On any failure a
    compilable placeholder stub is written alongside the preserved original source.
    """

    def __init__(self) -> None:
        self._gemini = GeminiClient()
        self._ollama = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    def convert_file(
        self,
        gd_path:     Path,
        output_path: Path,
    ) -> GDScriptConversionResult:
        """Convert one GDScript file to Unity C#."""
        result = GDScriptConversionResult(
            source_path=gd_path,
            output_path=output_path,
            success=False,
        )

        try:
            source = gd_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.error = f"Cannot read {gd_path.name}: {exc}"
            return result

        if not source.strip():
            result.error = f"{gd_path.name} is empty — skipped."
            return result

        class_name = _gd_class_name(gd_path.stem)
        prompt = _GD_TO_UNITY_PROMPT.format(class_name=class_name, gd_source=source)

        # ── LLM call: Gemini first, Ollama fallback ───────────────────────────
        code: Optional[str] = None

        if self._gemini.api_key:
            raw = self._gemini.generate_with_agent("architecture", prompt)
            if raw and raw.strip():
                code = _strip_code_fences(raw) if "```" in raw else raw.strip()
                log.info("[GDScript] Gemini converted %s", gd_path.name)

        if not code:
            try:
                raw = self._ollama.generate(prompt)
                if raw and raw.strip():
                    code = _strip_code_fences(raw) if "```" in raw else raw.strip()
                    log.info("[GDScript] Ollama converted %s", gd_path.name)
            except Exception as exc:
                log.warning("[GDScript] Ollama failed for %s: %s", gd_path.name, exc)

        if not code:
            _write_gd_fallback(gd_path, source, output_path, "No LLM output")
            result.error = "No LLM output — fallback stub written"
            return result

        # ── Post-process ──────────────────────────────────────────────────────
        code = _fix_class_inheritance(code)
        code = _clean_placeholders(code)

        # ── Validation + error-fix pass ───────────────────────────────────────
        errors = GodotToUnityCSharpConverter._validate(code)
        if errors and self._gemini.api_key:
            fixed = self._gemini.generate_with_agent(
                "error_fixer",
                _ERROR_FIX_PROMPT.format(errors="\n".join(errors), code=code),
            )
            if fixed and fixed.strip():
                code = _strip_code_fences(fixed) if "```" in fixed else fixed.strip()
                code = _fix_class_inheritance(code)
                code = _clean_placeholders(code)
                errors = GodotToUnityCSharpConverter._validate(code)

        # ── Write output ──────────────────────────────────────────────────────
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            result.error = f"Cannot write {output_path}: {exc}"
            return result

        if errors:
            result.error = "; ".join(errors)
            log.warning("[GDScript] residual issues in %s: %s", gd_path.name, result.error)

        result.csharp  = code
        result.success = True
        log.info("gdscript converted  %s → %s", gd_path.name, output_path.name)
        return result
