"""interfaces.py

Abstract protocol definitions for the three conversion pipeline layers.

These are lightweight structural contracts — any class with matching method
signatures satisfies the protocol without explicit inheritance.  The purpose
is to document expected shapes and enable static-type checking; no runtime
enforcement is added.

Pipeline flow:
    BaseParser → raw source data
    BaseIRBuilder → engine-agnostic IR
    BaseProjectBuilder → target-engine project on disk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    """Generic result returned by any BaseProjectBuilder implementation."""

    success:     bool
    output_dir:  Path
    warnings:    List[str] = field(default_factory=list)
    error:       str = ""


# ---------------------------------------------------------------------------
# Layer 1 — Source Parser
# ---------------------------------------------------------------------------

class BaseParser(Protocol):
    """Reads a source file and returns raw, source-engine-specific data.

    Implementations must NOT embed target-engine knowledge.  The returned
    dict is passed directly to a BaseIRBuilder.
    """

    def parse(self, source_path: Path, **kwargs: Any) -> Dict[str, Any]:
        """Parse *source_path* and return a raw data dict.

        The dict shape is source-engine-specific but must at minimum
        contain a ``"nodes"`` key with a list of raw node dicts.
        """
        ...


# ---------------------------------------------------------------------------
# Layer 2 — IR Builder
# ---------------------------------------------------------------------------

class BaseIRBuilder(Protocol):
    """Converts source-specific raw data into engine-agnostic IR.

    Implementations must NOT embed target-engine class names or file
    format details.  Engine-specific hints may be attached as optional
    The authoritative IR fields (``node_type``, ``components``, ``transform``)
    must remain engine-agnostic.
    """

    def build(
        self,
        raw_data: Dict[str, Any],
        scene_name: str,
        source_file: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convert *raw_data* into a spec-compliant scene IR dict.

        The returned dict must contain at minimum:
            ``ir_version``, ``scene_id``, ``scene_name``, ``nodes``
        """
        ...

    def validate(self, ir: Dict[str, Any]) -> List[str]:
        """Run structural checks on *ir*.

        Returns a (possibly empty) list of non-fatal warning strings.
        Raises on blocking errors that would prevent generation.
        """
        ...


# ---------------------------------------------------------------------------
# Layer 3 — Project Builder
# ---------------------------------------------------------------------------

class BaseProjectBuilder(Protocol):
    """Generates a complete target-engine project folder from IR.

    Implementations must consume ONLY the engine-agnostic IR.  Any
    target-specific knowledge (class names, file formats, resource paths)
    lives exclusively in this layer.
    """

    def build_project(
        self,
        scene_ir: Dict[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> BuildResult:
        """Write the target project to *output_dir* and return a result.

        *scene_ir* is the dict produced by a BaseIRBuilder implementation.
        All target-specific translation happens here.
        """
        ...
