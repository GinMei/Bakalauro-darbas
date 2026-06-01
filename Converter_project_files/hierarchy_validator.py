"""hierarchy_validator.py

Post-conversion structural validation for the Unity ↔ Godot converter.

Compares source and target IR trees to verify that hierarchy is preserved
identically.  Every check produces a boolean pass/fail result plus a numeric
fidelity score so callers can decide whether to block output or warn only.

Public API
----------
validate_hierarchy(source_ir, target_ir) -> HierarchyValidationResult
    Compare two IR dicts and return a full report.

tree_sig(node) -> tuple
    Produce a hashable structural fingerprint of an IR node tree.

HierarchyValidationResult
    Dataclass returned by validate_hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HierarchyValidationResult:
    """Full result of a hierarchy comparison between source and target IR."""

    # True when tree signatures match exactly (no drift at all).
    structural_match: bool

    # Percentage of nodes whose name, kind, and sibling-order match.
    # Range 0.0–100.0.  100.0 = perfect fidelity.
    fidelity_score: float

    # List of (node_path, expected_description, got_description) for every
    # mismatch found during tree traversal.
    drift_nodes: List[Tuple[str, str, str]] = field(default_factory=list)

    # ir_asset_ids referenced by the target but absent from its asset_registry.
    broken_refs: List[str] = field(default_factory=list)

    # True when every prefab_instance in source has a corresponding
    # prefab_instance (not a flattened group) in target.
    instance_integrity: bool = True

    # True when ir_doc_kind is the same in source and target.
    doc_kind_preserved: bool = True

    # Human-readable summary of all failures (empty string on full pass).
    summary: str = ""


# ---------------------------------------------------------------------------
# Structural fingerprint
# ---------------------------------------------------------------------------

def tree_sig(node: Dict[str, Any]) -> tuple:
    """Return a hashable structural fingerprint of *node* and its subtree.

    The signature captures:
      - node name
      - ir_node_kind (regular / scene_root / prefab_root / instance_node)
      - source_path of instance_ref (for instance_node nodes)
      - ordered child signatures

    Transforms, components, and metadata are intentionally excluded so the
    validator focuses solely on structural fidelity.
    """
    kind = node.get("ir_node_kind", "regular")
    ref_path = ""
    if kind == "instance_node":
        ref = node.get("instance_ref") or {}
        ref_path = ref.get("source_path", "") or ref.get("source_guid", "")
        if not ref_path:
            legacy = (node.get("components") or {}).get("instance_ref") or {}
            ref_path = legacy.get("source_res_path", "")

    children = sorted(
        node.get("children") or [],
        key=lambda c: c.get("child_index", 0),
    )
    return (
        node.get("node_name") or node.get("name", ""),
        kind,
        ref_path,
        tuple(tree_sig(c) for c in children if isinstance(c, dict)),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_nodes(nodes: List[Dict[str, Any]]) -> int:
    total = 0
    for n in nodes:
        if isinstance(n, dict):
            total += 1
            total += _count_nodes(n.get("children") or [])
    return total


def _find_drift(
    src_nodes: List[Dict[str, Any]],
    tgt_nodes: List[Dict[str, Any]],
    path: str = "",
) -> Tuple[List[Tuple[str, str, str]], int, int]:
    """Recursively compare two node lists.

    Returns (drift_entries, matching_count, total_count).
    """
    drift: List[Tuple[str, str, str]] = []
    matching = 0
    total = 0

    for idx, (sn, tn) in enumerate(zip(src_nodes, tgt_nodes)):
        if not (isinstance(sn, dict) and isinstance(tn, dict)):
            continue
        total += 1
        s_name = sn.get("node_name") or sn.get("name", f"_node_{idx}")
        t_name = tn.get("node_name") or tn.get("name", f"_node_{idx}")
        node_path = f"{path}/{s_name}" if path else s_name

        s_kind = sn.get("ir_node_kind", "regular")
        t_kind = tn.get("ir_node_kind", "regular")

        if s_name != t_name:
            drift.append((node_path, f"name={s_name!r}", f"name={t_name!r}"))
        elif s_kind != t_kind:
            drift.append((node_path, f"kind={s_kind}", f"kind={t_kind}"))
        else:
            matching += 1

        # Recurse into children
        s_children = sorted(sn.get("children") or [], key=lambda c: c.get("child_index", 0))
        t_children = sorted(tn.get("children") or [], key=lambda c: c.get("child_index", 0))

        if len(s_children) != len(t_children):
            drift.append((
                node_path,
                f"{len(s_children)} children",
                f"{len(t_children)} children",
            ))

        sub_drift, sub_match, sub_total = _find_drift(s_children, t_children, node_path)
        drift.extend(sub_drift)
        matching += sub_match
        total += sub_total

    # Nodes present in source but absent in target
    if len(src_nodes) > len(tgt_nodes):
        for extra in src_nodes[len(tgt_nodes):]:
            if isinstance(extra, dict):
                extra_name = extra.get("node_name") or extra.get("name", "?")
                drift.append((f"{path}/{extra_name}", "present", "MISSING"))
                total += 1

    # Nodes present in target but absent in source
    if len(tgt_nodes) > len(src_nodes):
        for extra in tgt_nodes[len(src_nodes):]:
            if isinstance(extra, dict):
                extra_name = extra.get("node_name") or extra.get("name", "?")
                drift.append((f"{path}/{extra_name}", "MISSING", "extra node"))

    return drift, matching, total


def _check_instance_integrity(
    src_nodes: List[Dict[str, Any]],
    tgt_nodes: List[Dict[str, Any]],
) -> List[str]:
    """Return names of nodes that were prefab_instance in source but not in target."""
    violations: List[str] = []

    for sn, tn in zip(src_nodes, tgt_nodes):
        if not (isinstance(sn, dict) and isinstance(tn, dict)):
            continue
        s_kind = sn.get("ir_node_kind", "regular")
        t_kind = tn.get("ir_node_kind", "regular")
        if s_kind == "instance_node" and t_kind != "instance_node":
            name = sn.get("node_name") or sn.get("name", "?")
            violations.append(f"'{name}': was instance_node, became {t_kind!r}")

        sub = _check_instance_integrity(
            sorted(sn.get("children") or [], key=lambda c: c.get("child_index", 0)),
            sorted(tn.get("children") or [], key=lambda c: c.get("child_index", 0)),
        )
        violations.extend(sub)

    return violations


def _check_broken_refs(
    nodes: List[Dict[str, Any]],
    registry: Dict[str, Any],
) -> List[str]:
    """Return ir_asset_ids referenced by instance_ref nodes but absent from registry."""
    broken: List[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ref = n.get("instance_ref") or {}
        if ref:
            aid = ref.get("ir_asset_id", "")
            if aid and aid not in registry:
                broken.append(aid)
        broken.extend(_check_broken_refs(n.get("children") or [], registry))
    return broken


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_hierarchy(
    source_ir: Dict[str, Any],
    target_ir: Dict[str, Any],
) -> HierarchyValidationResult:
    """Compare *source_ir* and *target_ir* for structural equivalence.

    Both arguments must be scene IR dicts as produced by unity_parser or
    godot_scene_parser (containing at minimum a ``"nodes"`` list).

    Returns a :class:`HierarchyValidationResult` with all fidelity metrics.
    A ``fidelity_score`` of 100.0 and ``structural_match = True`` means the
    hierarchies are identical.
    """
    src_nodes = source_ir.get("nodes") or []
    tgt_nodes = target_ir.get("nodes") or []
    tgt_registry = target_ir.get("asset_registry") or {}

    # ── Structural signature comparison ──────────────────────────────────────
    def _sig_list(nodes: List[Dict[str, Any]]) -> Optional[tuple]:
        valid = [n for n in nodes if isinstance(n, dict)]
        if not valid:
            return None
        if len(valid) == 1:
            return tree_sig(valid[0])
        return tuple(tree_sig(n) for n in valid)

    src_sig = _sig_list(src_nodes)
    tgt_sig = _sig_list(tgt_nodes)
    structural_match = (src_sig == tgt_sig)

    # ── Node-level drift ─────────────────────────────────────────────────────
    drift, matching, total = _find_drift(src_nodes, tgt_nodes)

    fidelity_score = (matching / total * 100.0) if total > 0 else 100.0

    # ── Instance integrity ───────────────────────────────────────────────────
    instance_violations = _check_instance_integrity(src_nodes, tgt_nodes)
    instance_integrity = len(instance_violations) == 0

    # ── Broken cross-file references ─────────────────────────────────────────
    broken_refs = _check_broken_refs(tgt_nodes, tgt_registry)

    # ── ir_doc_kind preservation ─────────────────────────────────────────────
    src_kind = source_ir.get("ir_doc_kind", "scene")
    tgt_kind = target_ir.get("ir_doc_kind", "scene")
    doc_kind_preserved = (src_kind == tgt_kind)

    # ── Summary string ───────────────────────────────────────────────────────
    parts: List[str] = []
    if not structural_match:
        parts.append(f"structural mismatch (score={fidelity_score:.1f}%)")
    if drift:
        parts.append(f"{len(drift)} drift node(s)")
    if instance_violations:
        parts.append(f"{len(instance_violations)} instance integrity violation(s)")
    if broken_refs:
        parts.append(f"{len(broken_refs)} broken asset reference(s)")
    if not doc_kind_preserved:
        parts.append(f"ir_doc_kind changed: {src_kind!r} → {tgt_kind!r}")
    summary = "; ".join(parts)

    return HierarchyValidationResult(
        structural_match=structural_match,
        fidelity_score=round(fidelity_score, 2),
        drift_nodes=drift,
        broken_refs=list(dict.fromkeys(broken_refs)),   # deduplicated, ordered
        instance_integrity=instance_integrity,
        doc_kind_preserved=doc_kind_preserved,
        summary=summary,
    )
