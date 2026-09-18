"""Tests for the URL frontier."""

from app.crawler.frontier import URLFrontier


def test_add_and_pop():
    f = URLFrontier()
    assert f.add("https://example.com", 0)
    t = f.pop()
    assert t is not None
    assert t.url == "https://example.com"
    assert t.depth == 0


def test_fifo_order():
    f = URLFrontier()
    f.add("https://a.com", 0)
    f.add("https://b.com", 1)
    f.add("https://c.com", 1)
    assert f.pop().url == "https://a.com"
    assert f.pop().url == "https://b.com"
    assert f.pop().url == "https://c.com"


def test_breadth_first_ordering():
    """BFS: depth-0 before depth-1 before depth-2."""
    f = URLFrontier()
    f.add("https://a.com", 0)
    f.add("https://b.com", 1)
    f.add("https://c.com", 1)
    f.add("https://d.com", 2)
    results = [f.pop().depth for _ in range(4)]
    assert results == [0, 1, 1, 2]


def test_duplicate_prevention():
    f = URLFrontier()
    assert f.add("https://example.com", 0) is True
    assert f.add("https://example.com", 1) is False
    assert len(f) == 1


def test_depth_preserved():
    f = URLFrontier()
    f.add("https://a.com", 3)
    t = f.pop()
    assert t.depth == 3


def test_empty_frontier():
    f = URLFrontier()
    assert f.empty() is True
    assert f.pop() is None
    assert len(f) == 0


def test_repeated_add_returns_false():
    f = URLFrontier()
    f.add("https://a.com", 0)
    assert f.add("https://a.com", 0) is False
    assert f.add("https://a.com", 1) is False


def test_queue_size():
    f = URLFrontier()
    f.add("https://a.com", 0)
    f.add("https://b.com", 0)
    assert len(f) == 2
    f.pop()
    assert len(f) == 1


def test_seen_count():
    f = URLFrontier()
    f.add("https://a.com", 0)
    f.add("https://b.com", 0)
    f.add("https://a.com", 0)  # duplicate
    assert f.seen_count == 2


def test_pop_empties_queue():
    f = URLFrontier()
    f.add("https://a.com", 0)
    f.pop()
    assert f.empty() is True
    assert f.pop() is None
