"""tests/test_dataset.py — Unit and integration tests for Day 7 dataset tooling."""

from __future__ import annotations

from complexity_lens.dataset import (
    ALLOWED_LABELS,
    ALLOWED_TAGS,
    ALLOWED_UNKNOWN_REASONS,
    compute_comment_stripped_sha256,
    compute_source_sha256,
    generate_dataset_report,
    load_dataset,
)


class TestSourceHashingPolicy:
    """Test the exact source hashing policy specified in Day 7 requirements."""

    def test_crlf_lf_equivalence(self):
        """CRLF and LF versions of identical source must yield identical SHA-256."""
        src_lf = "int main() {\n    return 0;\n}\n"
        src_crlf = "int main() {\r\n    return 0;\r\n}\r\n"
        hash_lf = compute_source_sha256(src_lf)
        hash_crlf = compute_source_sha256(src_crlf)
        assert hash_lf == hash_crlf
        assert len(hash_lf) == 64

    def test_trailing_whitespace_trimmed_from_end(self):
        """Trailing whitespace at the end of source is trimmed; whitespace in body is preserved."""
        src_clean = "int main() {\n    return 0;\n}"
        src_trailing_spaces = "int main() {\n    return 0;\n}   \n\t\n  "
        assert compute_source_sha256(src_clean) == compute_source_sha256(src_trailing_spaces)

    def test_intermediate_whitespace_preserved(self):
        """Trailing whitespace on intermediate lines is preserved."""
        src1 = "int main() {   \n    return 0;\n}"
        src2 = "int main() {\n    return 0;\n}"
        assert compute_source_sha256(src1) != compute_source_sha256(src2)

    def test_meaningful_source_change_changes_hash(self):
        """Altering source code statements must change the hash."""
        src1 = "int f(int x) { return x + 1; }"
        src2 = "int f(int x) { return x + 2; }"
        assert compute_source_sha256(src1) != compute_source_sha256(src2)

    def test_comment_stripped_hash(self):
        """Comment-stripped hashing ignores line and block comments."""
        src_with_comments = "// Header comment\nint f() { /* inline */ return 0; }\n"
        src_without_comments = "\nint f() {  return 0; }\n"
        h1 = compute_comment_stripped_sha256(src_with_comments)
        h2 = compute_comment_stripped_sha256(src_without_comments)
        assert h1 == h2


class TestDatasetLoaderAndIntegrity:
    """Validate dataset structure, family metadata, and program records."""

    def test_load_dataset_success(self):
        records, families, errors = load_dataset("datasets")
        assert len(errors) == 0, f"Unexpected dataset errors: {errors}"
        assert len(families) == 10
        assert len(records) == 38

    def test_record_schema_fields(self):
        records, _, errors = load_dataset("datasets")
        assert len(errors) == 0
        for r in records:
            d = r.to_dict()
            assert "program_id" in d and isinstance(d["program_id"], str)
            assert "family_id" in d and isinstance(d["family_id"], str)
            assert "source" in d and len(d["source"]) > 0
            assert "expected_label" in d and d["expected_label"] in ALLOWED_LABELS
            assert "label_rationale" in d and len(d["label_rationale"]) > 0
            assert d["label_status"] == "author_labeled"
            assert d["verified"] is False
            assert "source_sha256" in d and len(d["source_sha256"]) == 64
            # Must NOT contain train/validation/test split fields
            assert "split" not in d
            assert "split_group" not in d

    def test_family_metadata_fields(self):
        _, families, errors = load_dataset("datasets")
        assert len(errors) == 0
        for f in families:
            d = f.to_dict()
            assert d["label"] in ALLOWED_LABELS
            assert d["tag"] in ALLOWED_TAGS
            assert d["expected_parse"] in ("valid", "has_unsupported_constructs")
            assert len(d["programs"]) >= 3
            assert len(d["assumptions"]) > 0

    def test_unknown_programs_have_valid_reasons(self):
        records, _, errors = load_dataset("datasets")
        assert len(errors) == 0
        unknown_records = [r for r in records if r.expected_label == "UNKNOWN"]
        assert len(unknown_records) >= 5
        for r in unknown_records:
            assert r.unknown_reason in ALLOWED_UNKNOWN_REASONS
            assert r.unknown_explanation and len(r.unknown_explanation) > 5

    def test_rule_hard_proportion(self):
        records, families, errors = load_dataset("datasets")
        assert len(errors) == 0
        rule_hard_fam_ids = {f.family_id for f in families if f.tag == "rule_hard"}
        rule_hard_records = [r for r in records if r.family_id in rule_hard_fam_ids]
        # At least one third must be rule-hard
        assert len(rule_hard_records) >= len(records) / 3

    def test_no_duplicate_program_ids_or_hashes(self):
        records, _, errors = load_dataset("datasets")
        assert len(errors) == 0
        ids = [r.program_id for r in records]
        hashes = [r.source_sha256 for r in records]
        assert len(ids) == len(set(ids))
        assert len(hashes) == len(set(hashes))

    def test_missing_program_file_detected(self, tmp_path):
        fam_dir = tmp_path / "families"
        fam_dir.mkdir()
        prog_dir = tmp_path / "programs"
        prog_dir.mkdir()

        (fam_dir / "fam_test.yaml").write_text(
            "family_idea: test\nlabel: O1\nlabel_rationale: test\ntag: easy\n"
            "expected_parse: valid\nprograms:\n  - missing_prog\nassumptions:\n  - none\n"
        )
        (prog_dir / "fam_test").mkdir()
        recs, fams, errors = load_dataset(tmp_path)
        assert any("Missing program source file" in e for e in errors)

    def test_extra_unreferenced_file_detected(self, tmp_path):
        fam_dir = tmp_path / "families"
        fam_dir.mkdir()
        prog_dir = tmp_path / "programs"
        prog_dir.mkdir()

        (fam_dir / "fam_test.yaml").write_text(
            "family_idea: test\nlabel: O1\nlabel_rationale: test\ntag: easy\n"
            "expected_parse: valid\nprograms:\n  - prog1\nassumptions:\n  - none\n"
        )
        (prog_dir / "fam_test").mkdir()
        (prog_dir / "fam_test" / "prog1.cpp").write_text("int f() { return 1; }")
        (prog_dir / "fam_test" / "unreferenced.cpp").write_text("int g() { return 2; }")

        recs, fams, errors = load_dataset(tmp_path)
        assert any("Extra unreferenced source file" in e for e in errors)


class TestDatasetReport:
    """Validate comprehensive dataset reporting."""

    def test_generate_report_structure(self):
        report = generate_dataset_report("datasets", check_parser=True)
        assert report["total_records"] == 38
        assert report["total_families"] == 10
        assert len(report["metadata_schema_failures"]) == 0
        assert len(report["duplicate_program_ids"]) == 0
        assert len(report["duplicate_source_hashes"]) == 0
        assert report["parser_validity"]["status"] in ("PARSER_AVAILABLE", "PARSER_UNAVAILABLE")
