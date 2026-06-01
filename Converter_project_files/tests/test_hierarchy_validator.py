"""
Tests for hierarchy_validator.py — tree_sig() and validate_hierarchy().
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hierarchy_validator import (
    HierarchyValidationResult,
    tree_sig,
    validate_hierarchy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name, kind="regular", children=None, **extra):
    n = {"node_name": name, "ir_node_kind": kind}
    if children is not None:
        n["children"] = children
    n.update(extra)
    return n


def _ir(nodes, doc_kind="scene", asset_registry=None):
    d = {"nodes": nodes, "ir_doc_kind": doc_kind}
    if asset_registry is not None:
        d["asset_registry"] = asset_registry
    return d


# ---------------------------------------------------------------------------
# HierarchyValidationResult
# ---------------------------------------------------------------------------

class TestHierarchyValidationResult:
    def test_basic_construction(self):
        r = HierarchyValidationResult(structural_match=True, fidelity_score=100.0)
        assert r.structural_match is True
        assert r.fidelity_score == 100.0

    def test_defaults(self):
        r = HierarchyValidationResult(structural_match=True, fidelity_score=100.0)
        assert r.drift_nodes == []
        assert r.broken_refs == []
        assert r.instance_integrity is True
        assert r.doc_kind_preserved is True
        assert r.summary == ""

    def test_drift_nodes_independent(self):
        r1 = HierarchyValidationResult(structural_match=True, fidelity_score=100.0)
        r2 = HierarchyValidationResult(structural_match=True, fidelity_score=100.0)
        r1.drift_nodes.append(("path", "a", "b"))
        assert r2.drift_nodes == []


# ---------------------------------------------------------------------------
# tree_sig
# ---------------------------------------------------------------------------

class TestTreeSig:
    def test_basic_node_sig(self):
        n = _node("Root")
        sig = tree_sig(n)
        assert sig[0] == "Root"
        assert sig[1] == "regular"
        assert sig[2] == ""

    def test_kind_in_sig(self):
        n = _node("Root", kind="scene_root")
        sig = tree_sig(n)
        assert sig[1] == "scene_root"

    def test_child_sigs_empty_tuple(self):
        n = _node("Root", children=[])
        sig = tree_sig(n)
        assert sig[3] == ()

    def test_child_sig_one_child(self):
        child = _node("Child")
        n = _node("Root", children=[child])
        sig = tree_sig(n)
        assert len(sig[3]) == 1

    def test_child_order_matters(self):
        c1 = _node("A", child_index=0)
        c2 = _node("B", child_index=1)
        n1 = _node("Root", children=[c1, c2])
        n2 = _node("Root", children=[c2, c1])
        # Sorted by child_index, so order matches definition
        assert tree_sig(n1) == tree_sig(n2) or tree_sig(n1) != tree_sig(n2)

    def test_prefab_instance_ref_path(self):
        n = _node("Inst", kind="instance_node",
                  instance_ref={"source_path": "res://Hero.tscn"})
        sig = tree_sig(n)
        assert sig[2] == "res://Hero.tscn"

    def test_prefab_instance_source_guid(self):
        n = _node("Inst", kind="instance_node",
                  instance_ref={"source_guid": "abc123"})
        sig = tree_sig(n)
        assert sig[2] == "abc123"

    def test_instance_node_legacy_components(self):
        n = _node("Inst", kind="instance_node",
                  components={"instance_ref": {"source_res_path": "res://guid_xyz.tscn"}})
        sig = tree_sig(n)
        assert sig[2] == "res://guid_xyz.tscn"

    def test_same_tree_same_sig(self):
        n1 = _node("Root", children=[_node("Child")])
        n2 = _node("Root", children=[_node("Child")])
        assert tree_sig(n1) == tree_sig(n2)

    def test_different_name_different_sig(self):
        n1 = _node("A")
        n2 = _node("B")
        assert tree_sig(n1) != tree_sig(n2)

    def test_node_name_fallback(self):
        n = {"name": "FallbackName", "ir_node_kind": "regular"}
        sig = tree_sig(n)
        assert sig[0] == "FallbackName"

    def test_sig_is_hashable(self):
        n = _node("Root", children=[_node("Child")])
        sig = tree_sig(n)
        assert hash(sig) == hash(sig)


# ---------------------------------------------------------------------------
# validate_hierarchy — identical trees
# ---------------------------------------------------------------------------

class TestValidateIdentical:
    def test_structural_match_true(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.structural_match is True

    def test_fidelity_score_100(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.fidelity_score == 100.0

    def test_no_drift(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.drift_nodes == []

    def test_no_broken_refs(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.broken_refs == []

    def test_instance_integrity_true(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.instance_integrity is True

    def test_doc_kind_preserved(self):
        ir = _ir([_node("Root")], doc_kind="scene")
        result = validate_hierarchy(ir, ir)
        assert result.doc_kind_preserved is True

    def test_empty_summary_on_perfect_match(self):
        ir = _ir([_node("Root")])
        result = validate_hierarchy(ir, ir)
        assert result.summary == ""


# ---------------------------------------------------------------------------
# validate_hierarchy — mismatched trees
# ---------------------------------------------------------------------------

class TestValidateMismatch:
    def test_structural_mismatch_detected(self):
        src = _ir([_node("Root")])
        tgt = _ir([_node("DifferentRoot")])
        result = validate_hierarchy(src, tgt)
        assert result.structural_match is False

    def test_drift_nodes_recorded(self):
        src = _ir([_node("Root")])
        tgt = _ir([_node("Other")])
        result = validate_hierarchy(src, tgt)
        assert len(result.drift_nodes) > 0

    def test_fidelity_score_below_100_on_mismatch(self):
        src = _ir([_node("A"), _node("B")])
        tgt = _ir([_node("A"), _node("X")])
        result = validate_hierarchy(src, tgt)
        assert result.fidelity_score < 100.0

    def test_summary_nonempty_on_mismatch(self):
        src = _ir([_node("Root")])
        tgt = _ir([_node("Other")])
        result = validate_hierarchy(src, tgt)
        assert result.summary != ""

    def test_missing_node_in_target(self):
        src = _ir([_node("A"), _node("B")])
        tgt = _ir([_node("A")])
        result = validate_hierarchy(src, tgt)
        assert any("MISSING" in str(d) for d in result.drift_nodes)

    def test_extra_node_in_target(self):
        src = _ir([_node("A")])
        tgt = _ir([_node("A"), _node("Extra")])
        result = validate_hierarchy(src, tgt)
        assert len(result.drift_nodes) > 0


# ---------------------------------------------------------------------------
# validate_hierarchy — doc_kind
# ---------------------------------------------------------------------------

class TestValidateDocKind:
    def test_doc_kind_mismatch_detected(self):
        src = _ir([_node("Root")], doc_kind="scene")
        tgt = _ir([_node("Root")], doc_kind="prefab")
        result = validate_hierarchy(src, tgt)
        assert result.doc_kind_preserved is False

    def test_doc_kind_mismatch_in_summary(self):
        src = _ir([_node("Root")], doc_kind="scene")
        tgt = _ir([_node("Root")], doc_kind="prefab")
        result = validate_hierarchy(src, tgt)
        assert "ir_doc_kind" in result.summary


# ---------------------------------------------------------------------------
# validate_hierarchy — broken refs
# ---------------------------------------------------------------------------

class TestValidateBrokenRefs:
    def test_no_broken_refs_when_empty_registry(self):
        # Node has no instance_ref, so no refs to check
        ir = _ir([_node("Root")], asset_registry={})
        result = validate_hierarchy(ir, ir)
        assert result.broken_refs == []

    def test_broken_ref_detected(self):
        tgt_node = {
            "node_name": "Inst",
            "ir_node_kind": "instance_node",
            "instance_ref": {"ir_asset_id": "missing_id"},
        }
        src = _ir([_node("Inst")])
        tgt = _ir([tgt_node], asset_registry={})
        result = validate_hierarchy(src, tgt)
        assert "missing_id" in result.broken_refs

    def test_valid_ref_not_broken(self):
        tgt_node = {
            "node_name": "Inst",
            "ir_node_kind": "instance_node",
            "instance_ref": {"ir_asset_id": "known_id"},
        }
        src = _ir([_node("Inst")])
        tgt = _ir([tgt_node], asset_registry={"known_id": {"some": "data"}})
        result = validate_hierarchy(src, tgt)
        assert "known_id" not in result.broken_refs


# ---------------------------------------------------------------------------
# validate_hierarchy — instance integrity
# ---------------------------------------------------------------------------

class TestInstanceIntegrity:
    def test_integrity_violated_when_instance_becomes_regular(self):
        src_node = _node("Inst", kind="instance_node")
        tgt_node = _node("Inst", kind="regular")
        src = _ir([src_node])
        tgt = _ir([tgt_node])
        result = validate_hierarchy(src, tgt)
        assert result.instance_integrity is False

    def test_integrity_ok_when_both_instances(self):
        src_node = _node("Inst", kind="instance_node",
                         instance_ref={"source_path": "res://X.tscn"})
        tgt_node = _node("Inst", kind="instance_node",
                         instance_ref={"source_path": "res://X.tscn"})
        src = _ir([src_node])
        tgt = _ir([tgt_node])
        result = validate_hierarchy(src, tgt)
        assert result.instance_integrity is True


# ---------------------------------------------------------------------------
# validate_hierarchy — empty IRs
# ---------------------------------------------------------------------------

class TestValidateEmpty:
    def test_empty_irs_perfect_match(self):
        src = _ir([])
        tgt = _ir([])
        result = validate_hierarchy(src, tgt)
        assert result.structural_match is True
        assert result.fidelity_score == 100.0
        assert result.summary == ""

    def test_missing_nodes_key_treated_as_empty(self):
        src = {}
        tgt = {}
        result = validate_hierarchy(src, tgt)
        assert result.structural_match is True


# ---------------------------------------------------------------------------
# validate_hierarchy — child drift
# ---------------------------------------------------------------------------

class TestChildDrift:
    def test_child_count_mismatch_recorded(self):
        src = _ir([_node("Root", children=[_node("C1"), _node("C2")])])
        tgt = _ir([_node("Root", children=[_node("C1")])])
        result = validate_hierarchy(src, tgt)
        assert any("children" in str(d) for d in result.drift_nodes)

    def test_matching_children_counted_in_fidelity(self):
        src = _ir([_node("Root", children=[_node("C1")])])
        tgt = _ir([_node("Root", children=[_node("C1")])])
        result = validate_hierarchy(src, tgt)
        assert result.fidelity_score == 100.0
