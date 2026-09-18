import pytest
from app.crawler.url_normalizer import (
    normalize_url,
    is_valid_url,
    extract_domain,
    is_same_domain,
)

def test_absolute_url_normalized():
    assert normalize_url("http://example.com") == "http://example.com/"

def test_relative_url_resolved():
    assert normalize_url("/about", "http://example.com/") == "http://example.com/about"
    assert normalize_url("about", "http://example.com/") == "http://example.com/about"

def test_parent_path_resolved():
    assert normalize_url("../contact", "http://example.com/a/b/") == "http://example.com/a/contact"

def test_query_string_preserved():
    assert normalize_url("http://example.com/?q=test&page=1") == "http://example.com/?q=test&page=1"

def test_fragment_removed():
    assert normalize_url("http://example.com/#section-1") == "http://example.com/"
    assert normalize_url("http://example.com/page#top") == "http://example.com/page"

def test_trailing_slash_preserved():
    assert normalize_url("http://example.com/dir/") == "http://example.com/dir/"

def test_uppercase_scheme_lowered():
    assert normalize_url("HTTP://Example.COM") == "http://example.com/"

def test_unsupported_scheme_javascript():
    assert normalize_url("javascript:alert(1)") is None

def test_unsupported_scheme_file():
    assert normalize_url("file:///etc/passwd") is None

def test_unsupported_scheme_data():
    assert normalize_url("data:text/html;base64,PHN2Zy9vbmxvYWQ9YWxlcnQoMik+") is None

def test_unsupported_scheme_ftp():
    assert normalize_url("ftp://example.com/file.txt") is None

def test_malformed_url():
    assert normalize_url("http://") is None

def test_empty_url():
    assert normalize_url("") is None
    assert normalize_url("   ") is None

def test_whitespace_url():
    assert normalize_url(" \t \n") is None

def test_duplicate_normalization():
    first = normalize_url("http://example.com/#top")
    second = normalize_url(first)
    assert first == second == "http://example.com/"

def test_extract_domain():
    assert extract_domain("http://example.com/") == "example.com"
    assert extract_domain("https://sub.example.com/path") == "sub.example.com"

def test_extract_domain_strips_www():
    assert extract_domain("http://www.example.com/") == "example.com"

def test_is_same_domain_true():
    assert is_same_domain("http://example.com/page1", "http://example.com/page2") is True

def test_is_same_domain_false():
    assert is_same_domain("http://example.com/page1", "http://other.com/page2") is False

def test_is_same_domain_www():
    assert is_same_domain("http://www.example.com/", "http://example.com/") is True

def test_localhost_rejected():
    assert normalize_url("http://localhost:8080/test") is None

def test_127_0_0_1_rejected():
    assert normalize_url("http://127.0.0.1/test") is None

def test_0_0_0_0_rejected():
    assert normalize_url("http://0.0.0.0/test") is None

def test_private_ip_rejected():
    assert normalize_url("http://192.168.1.1/admin") is None
    assert normalize_url("http://10.0.0.5/admin") is None

def test_is_valid_url_true():
    assert is_valid_url("http://example.com") is True

def test_is_valid_url_false():
    assert is_valid_url("javascript:alert(1)") is False
