"""Test swedefend corpus generation."""

from swedefend.corpus import generate_corpus


def test_corpus_size():
    """Verify corpus generates at least 60 cases."""
    corpus = generate_corpus()
    assert len(corpus) >= 60, f"Expected >=60 cases, got {len(corpus)}"


def test_corpus_balance():
    """Verify mix of malicious and benign cases."""
    corpus = generate_corpus()
    malicious = [c for c in corpus if c.is_malicious]
    benign = [c for c in corpus if not c.is_malicious]
    assert len(malicious) > 0, "No malicious cases"
    assert len(benign) > 0, "No benign cases"
    assert len(malicious) >= 40, f"Expected >=40 malicious, got {len(malicious)}"
    assert len(benign) >= 10, f"Expected >=10 benign, got {len(benign)}"


def test_cwe_coverage():
    """Verify all 5 CWE families are present."""
    corpus = generate_corpus()
    cwes = {c.cwe_type for c in corpus if c.cwe_type}
    expected = {"CWE-78", "CWE-502", "CWE-327", "CWE-94", "CWE-22"}
    assert cwes == expected, f"Missing CWEs: {expected - cwes}"


def test_corpus_structure():
    """Verify each case has required fields."""
    corpus = generate_corpus()
    for case in corpus[:5]:  # sample
        assert case.name
        assert case.issue_text
        assert case.patch_source
        assert isinstance(case.is_malicious, bool)
        assert case.description
