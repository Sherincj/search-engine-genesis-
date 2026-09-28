"""Tests for the query processor."""

from app.search.query_processor import QueryProcessor

def test_empty_query():
    qp = QueryProcessor()
    res = qp.process("")
    assert res.is_empty
    assert res.terms == ()
    assert res.phrases == ()

def test_whitespace_query():
    qp = QueryProcessor()
    res = qp.process("   \n\t  ")
    assert res.is_empty
    assert res.terms == ()
    assert res.phrases == ()

def test_single_term():
    qp = QueryProcessor()
    res = qp.process("Python")
    assert len(res.terms) == 1
    assert res.terms[0].term == "python"
    assert res.terms[0].frequency == 1
    assert res.terms[0].weight == 1.0

def test_multiple_terms():
    qp = QueryProcessor()
    res = qp.process("python web development")
    assert len(res.terms) == 3
    assert res.terms[0].term == "python"
    assert res.terms[1].term == "web"
    assert res.terms[2].term == "development"
    for qt in res.terms:
        assert qt.frequency == 1
        assert qt.weight == 1.0

def test_repeated_terms():
    qp = QueryProcessor()
    res = qp.process("python python python development")
    assert len(res.terms) == 2
    
    assert res.terms[0].term == "python"
    assert res.terms[0].frequency == 3
    assert res.terms[0].weight > 1.0
    
    assert res.terms[1].term == "development"
    assert res.terms[1].frequency == 1
    assert res.terms[1].weight == 1.0

def test_repeated_terms_ordering():
    qp = QueryProcessor()
    # "development" appears first
    res = qp.process("development python python")
    assert len(res.terms) == 2
    assert res.terms[0].term == "development"
    assert res.terms[1].term == "python"

def test_phrase_parsing():
    qp = QueryProcessor()
    res = qp.process('"python development"')
    assert len(res.phrases) == 1
    assert res.phrases[0].terms == ("python", "development")
    # The terms in the phrase are also recorded in `terms` for frequency scoring
    assert len(res.terms) == 2
    assert res.terms[0].term == "python"
    assert res.terms[1].term == "development"

def test_duplicate_phrases():
    qp = QueryProcessor()
    res = qp.process('"python development" "python development"')
    # Phrases should be deduplicated
    assert len(res.phrases) == 1
    assert res.phrases[0].terms == ("python", "development")
    # Terms should have frequency 2
    assert res.terms[0].frequency == 2
    assert res.terms[1].frequency == 2

def test_mixed_terms_and_phrases():
    qp = QueryProcessor()
    res = qp.process('learn "python development" fast')
    assert len(res.phrases) == 1
    assert res.phrases[0].terms == ("python", "development")
    
    assert len(res.terms) == 4
    # Expected order: learn, python, development, fast
    assert res.terms[0].term == "learn"
    assert res.terms[1].term == "python"
    assert res.terms[2].term == "development"
    assert res.terms[3].term == "fast"

def test_all_terms_property():
    qp = QueryProcessor()
    res = qp.process('python "web development" python')
    # Unique terms: python, web, development
    assert res.all_terms == ("python", "web", "development")


def test_punctuation_normalized():
    qp = QueryProcessor()
    res = qp.process("Python, development!")
    assert len(res.terms) == 2
    assert res.terms[0].term == "python"
    assert res.terms[1].term == "development"

def test_case_normalized():
    qp = QueryProcessor()
    res = qp.process("Python DEVELOPMENT")
    assert res.terms[0].term == "python"
    assert res.terms[1].term == "development"

def test_unicode_terms():
    qp = QueryProcessor()
    res = qp.process("café résumé")
    assert len(res.terms) == 2
    assert res.terms[0].term == "café"
    assert res.terms[1].term == "résumé"

def test_numbers_in_query():
    qp = QueryProcessor()
    res = qp.process("python 3 tutorial")
    terms = [qt.term for qt in res.terms]
    assert "python" in terms
    assert "3" in terms
    assert "tutorial" in terms

def test_unmatched_quotes():
    qp = QueryProcessor()
    # Unmatched quote should be treated as normal text
    res = qp.process('"python development')
    assert not res.is_empty
    assert len(res.phrases) == 0  # No valid phrase extracted

def test_original_query_preserved():
    qp = QueryProcessor()
    raw = "Python DEVELOPMENT"
    res = qp.process(raw)
    assert res.original_query == raw

def test_multiple_phrases():
    qp = QueryProcessor()
    res = qp.process('"python development" "web framework"')
    assert len(res.phrases) == 2
    assert res.phrases[0].terms == ("python", "development")
    assert res.phrases[1].terms == ("web", "framework")

def test_is_phrase_only():
    qp = QueryProcessor()
    res = qp.process('"python development"')
    assert res.is_phrase_only is True

def test_is_not_phrase_only_with_loose_terms():
    qp = QueryProcessor()
    res = qp.process('learn "python development"')
    assert res.is_phrase_only is False

def test_repeated_term_interleaved():
    """python development python → python freq=2, development freq=1."""
    qp = QueryProcessor()
    res = qp.process("python development python")
    assert len(res.terms) == 2
    assert res.terms[0].term == "python"
    assert res.terms[0].frequency == 2
    assert res.terms[1].term == "development"
    assert res.terms[1].frequency == 1
