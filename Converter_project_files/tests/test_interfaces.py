"""
Tests for interfaces.py — BuildResult dataclass and Protocol classes.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interfaces import BuildResult, BaseParser, BaseIRBuilder, BaseProjectBuilder


class TestBuildResult:
    def test_success_true(self):
        r = BuildResult(success=True, output_dir=Path("/tmp/out"))
        assert r.success is True

    def test_success_false(self):
        r = BuildResult(success=False, output_dir=Path("/tmp/out"))
        assert r.success is False

    def test_output_dir_stored(self):
        p = Path("/some/path")
        r = BuildResult(success=True, output_dir=p)
        assert r.output_dir == p

    def test_warnings_default_empty(self):
        r = BuildResult(success=True, output_dir=Path("/tmp"))
        assert r.warnings == []

    def test_error_default_empty_string(self):
        r = BuildResult(success=True, output_dir=Path("/tmp"))
        assert r.error == ""

    def test_warnings_provided(self):
        r = BuildResult(success=True, output_dir=Path("/tmp"), warnings=["warn1", "warn2"])
        assert r.warnings == ["warn1", "warn2"]

    def test_error_provided(self):
        r = BuildResult(success=False, output_dir=Path("/tmp"), error="something failed")
        assert r.error == "something failed"

    def test_warnings_independent_per_instance(self):
        r1 = BuildResult(success=True, output_dir=Path("/tmp"))
        r2 = BuildResult(success=True, output_dir=Path("/tmp"))
        r1.warnings.append("w")
        assert r2.warnings == []

    def test_dataclass_equality(self):
        p = Path("/tmp")
        r1 = BuildResult(success=True, output_dir=p)
        r2 = BuildResult(success=True, output_dir=p)
        assert r1 == r2

    def test_dataclass_inequality_on_success(self):
        p = Path("/tmp")
        r1 = BuildResult(success=True, output_dir=p)
        r2 = BuildResult(success=False, output_dir=p)
        assert r1 != r2


class TestProtocolsExist:
    def test_base_parser_is_protocol(self):
        from typing import Protocol
        assert issubclass(BaseParser, Protocol)

    def test_base_ir_builder_is_protocol(self):
        from typing import Protocol
        assert issubclass(BaseIRBuilder, Protocol)

    def test_base_project_builder_is_protocol(self):
        from typing import Protocol
        assert issubclass(BaseProjectBuilder, Protocol)

    def test_base_parser_has_parse_method(self):
        assert hasattr(BaseParser, "parse")

    def test_base_ir_builder_has_build_method(self):
        assert hasattr(BaseIRBuilder, "build")

    def test_base_ir_builder_has_validate_method(self):
        assert hasattr(BaseIRBuilder, "validate")

    def test_base_project_builder_has_build_project_method(self):
        assert hasattr(BaseProjectBuilder, "build_project")


class TestBuildResultRepr:
    def test_repr_contains_success(self):
        r = BuildResult(success=True, output_dir=Path("/tmp"))
        assert "True" in repr(r)

    def test_repr_contains_error(self):
        r = BuildResult(success=False, output_dir=Path("/tmp"), error="oops")
        assert "oops" in repr(r)
