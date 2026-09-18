"""Tests for the HTML parser."""

from app.crawler.parser import parse_html


# ── Title extraction ─────────────────────────────────────────

def test_title_extracted():
    result = parse_html("<title>Example Page</title>")
    assert result.title == "Example Page"


def test_missing_title():
    result = parse_html("<html><body>No title</body></html>")
    assert result.title is None


def test_empty_title():
    result = parse_html("<title>   </title>")
    assert result.title is None


def test_multiple_titles():
    """Uses the first <title> tag."""
    html = "<title>First</title><title>Second</title>"
    result = parse_html(html)
    assert result.title == "First"


# ── Body text extraction ─────────────────────────────────────

def test_body_text_extracted():
    html = "<h1>Hello</h1><p>World</p>"
    result = parse_html(html)
    assert "Hello" in result.text
    assert "World" in result.text


def test_script_excluded():
    html = '<script>dangerous()</script><p>Safe text</p>'
    result = parse_html(html)
    assert "dangerous" not in result.text
    assert "Safe text" in result.text


def test_style_excluded():
    html = '<style>body { color: red; }</style><p>Content</p>'
    result = parse_html(html)
    assert "color" not in result.text
    assert "Content" in result.text


def test_noscript_excluded():
    html = '<noscript>Enable JS</noscript><p>Real</p>'
    result = parse_html(html)
    assert "Enable" not in result.text
    assert "Real" in result.text


def test_nav_excluded():
    html = '<nav>Menu</nav><p>Article</p>'
    result = parse_html(html)
    assert "Menu" not in result.text
    assert "Article" in result.text


def test_whitespace_normalized():
    html = '<p>Hello    World</p><p>  More  </p>'
    result = parse_html(html)
    assert "  " not in result.text
    assert "Hello" in result.text
    assert "World" in result.text


def test_empty_body():
    result = parse_html("<html></html>")
    assert result.text == ""


def test_malformed_html():
    html = '<div><p>Unclosed<span>tags'
    result = parse_html(html)
    assert "Unclosed" in result.text
    assert "tags" in result.text


# ── Link extraction ──────────────────────────────────────────

def test_links_extracted():
    html = '<a href="/about">About</a><a href="/contact">Contact</a>'
    result = parse_html(html)
    assert "/about" in result.links
    assert "/contact" in result.links


def test_javascript_link_ignored():
    html = '<a href="javascript:alert(1)">Bad</a><a href="/good">Good</a>'
    result = parse_html(html)
    assert "/good" in result.links
    assert len(result.links) == 1


def test_empty_href_ignored():
    html = '<a href="">Empty</a><a href="/page">Page</a>'
    result = parse_html(html)
    assert len(result.links) == 1
    assert result.links[0] == "/page"


def test_duplicate_links_removed():
    html = '<a href="/page">A</a><a href="/page">B</a>'
    result = parse_html(html)
    assert len(result.links) == 1


def test_multiple_links():
    html = '<a href="/a">A</a><a href="/b">B</a><a href="/c">C</a>'
    result = parse_html(html)
    assert len(result.links) == 3


def test_mailto_ignored():
    html = '<a href="mailto:user@example.com">Email</a>'
    result = parse_html(html)
    assert len(result.links) == 0


# ── Canonical URL ────────────────────────────────────────────

def test_canonical_extracted():
    html = '<link rel="canonical" href="https://example.com/page">'
    result = parse_html(html)
    assert result.canonical_url == "https://example.com/page"


def test_canonical_missing():
    result = parse_html("<html><body>No canonical</body></html>")
    assert result.canonical_url is None


def test_canonical_empty_href():
    html = '<link rel="canonical" href="">'
    result = parse_html(html)
    assert result.canonical_url is None


# ── Complex pages ────────────────────────────────────────────

def test_full_page():
    html = '''
    <html>
    <head><title>Python Guide</title></head>
    <body>
        <nav>Menu</nav>
        <h1>Learn Python</h1>
        <p>Python is a programming language.</p>
        <script>console.log('hi');</script>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
        <link rel="canonical" href="https://example.com/python">
    </body>
    </html>
    '''
    result = parse_html(html)
    assert result.title == "Python Guide"
    assert "Python is a programming language" in result.text
    assert "console" not in result.text
    assert "Menu" not in result.text
    assert "/about" in result.links
    assert "/contact" in result.links
    assert result.canonical_url == "https://example.com/python"
