# Snippet Generation

Phase 9 adds search-result snippets: short, highlighted excerpts showing
users the most relevant portion of each matching document.

## Purpose

Search engines display snippets so users can evaluate relevance without
reading entire documents.  A good snippet shows:

1. The region of the document most relevant to the query.
2. Highlighted query terms for quick scanning.
3. Enough context to understand the match.

## Architecture

The snippet system is cleanly separated from ranking:

```text
SearchService
    ├── SearchEngine (ranking — unchanged)
    └── SnippetGenerator (presentation)
```

The `SnippetGenerator` is **independent** of:
- FastAPI
- SQLAlchemy
- BM25 / ranking
- The inverted index

It operates entirely on raw strings and normalised query terms.

## Query Matching

The generator uses the **same Phase 3 tokenizer** for matching, ensuring
consistency between indexing, query parsing, and snippet generation:

```text
Document text: "Python Development Guide"
Query:         "python"
Normalised:    "python" matches "Python" (case-insensitive)
```

Matching is normalised but output preserves original casing.

## Field Selection

The generator follows this priority:

```text
1. Body containing query terms      → body snippet with highlights
2. Title containing query terms     → title snippet with highlights
3. Body fallback (no matches)       → first 160 chars of body
4. Title fallback (no matches)      → title text
5. No text available                → None
```

## Snippet Windowing

For long documents, the generator extracts a window (default 160 chars)
centered around the best match region.

### Dense Match Preference

When multiple query terms appear close together, that region is preferred
over isolated matches:

```text
Query: "python development"

Document:
"Python is a language. ... Many developers use Python for web development ..."

Preferred snippet:
"...Many developers use Python for web development..."
(both terms nearby)
```

### Word Boundaries

The window snaps to word boundaries to avoid cutting words mid-token.

### Ellipsis

Ellipsis (`...`) is added when content is truncated:
- Leading `...` when the window doesn't start at the beginning
- Trailing `...` when the window doesn't reach the end
- No ellipsis when the entire text fits within the window

## Highlighting

Matched terms are wrapped in `<mark>` tags in the HTML output:

```html
Learn <mark>Python</mark> programming and web <mark>development</mark>...
```

### Security

All document content is **HTML-escaped** before highlighting:

```text
Input:  Python & JavaScript <script>alert("x")</script>
Output: <mark>Python</mark> &amp; JavaScript &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;
```

This prevents:
- XSS attacks via injected `<script>` tags
- HTML injection via angle brackets
- Attribute injection via quotes

The `<mark>` tags are added **after** escaping by the trusted internal
routine, so they are the only unescaped HTML in the output.

## API Response

The search response includes a nullable `snippet` field:

```json
{
  "document_id": 1,
  "title": "Python Guide",
  "url": "https://example.com/python",
  "score": 4.72,
  "snippet": "Learn <mark>Python</mark> programming..."
}
```

The `snippet` field is `null` when no useful text exists.

## Configuration

| Setting              | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `SNIPPET_LENGTH`     | 160     | Maximum snippet length in characters     |
| `SNIPPET_MIN_LENGTH` | 80      | Minimum snippet length for expansion     |

## Edge Cases

| Case                    | Behavior                                    |
|-------------------------|---------------------------------------------|
| Empty query             | Returns `None`                              |
| Empty body and title    | Returns `None`                              |
| No query terms found    | Returns fallback snippet (no highlights)    |
| Title only              | Uses title for snippet                      |
| Body only               | Uses body for snippet                       |
| Unicode text            | Fully supported (café, 中文, 한국어)        |
| HTML in document text   | Safely escaped                              |

## Performance

Snippets are generated **only for the current page** of results.
For a `limit=20` query, at most 20 snippets are generated.
The generator makes a single pass over each document's text.
