"""Test swedefend corpus generation."""

from swedefend.corpus import generate_corpus


def test_corpus_size():
    """Verify corpus generates at least 140 cases (49 mal + 100+ benign)."""
    corpus = generate_corpus()
    assert len(corpus) >= 140, f"Expected >=140 cases, got {len(corpus)}"


def test_corpus_balance():
    """Verify mix of malicious and benign cases."""
    corpus = generate_corpus()
    malicious = [c for c in corpus if c.is_malicious]
    benign = [c for c in corpus if not c.is_malicious]
    assert len(malicious) > 0, "No malicious cases"
    assert len(benign) > 0, "No benign cases"
    assert len(malicious) >= 40, f"Expected >=40 malicious, got {len(malicious)}"
    # AI-Discovery D3 expansion: benign corpus has 100 diverse cases so the
    # Wilson upper bound on FPR at 0 FPs tightens from ~16% (n=20) to ~3.6%
    # (n=100).
    assert len(benign) >= 100, f"Expected >=100 benign, got {len(benign)}"


def test_benign_family_diversity():
    """AI-Discovery: benign cases span >=15 distinct scenario templates."""
    corpus = generate_corpus()
    families = {c.family for c in corpus if not c.is_malicious}
    assert len(families) >= 15, (
        f"Expected >=15 benign families for statistical diversity, got "
        f"{len(families)}: {sorted(families)}"
    )


def test_cwe_coverage():
    """Verify all 5 SWExploit CWE families are present in the corpus."""
    corpus = generate_corpus()
    cwes = {c.cwe_type for c in corpus if c.cwe_type}
    expected = {"CWE-78", "CWE-502", "CWE-327", "CWE-94", "CWE-22"}
    # The corpus additionally contains false-premise cases with different CWE
    # tags (e.g. CWE-284, CWE-352); those are welcome extras — we only require
    # the SWExploit-canonical five to be a subset.
    assert expected.issubset(cwes), f"Missing CWEs: {expected - cwes}"


def test_corpus_structure():
    """Verify each case has required fields."""
    corpus = generate_corpus()
    for case in corpus[:5]:  # sample
        assert case.name
        assert case.issue_text
        assert case.patch_source
        assert isinstance(case.is_malicious, bool)
        assert case.description
