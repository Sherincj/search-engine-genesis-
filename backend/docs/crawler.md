# Web Crawler

Phase 10 adds a breadth-first, same-domain web crawler that fetches HTML
pages, extracts content, stores documents in SQLite, and updates the
search index.

## Architecture

```text
POST /crawl
    ↓
Crawl API
    ↓
Crawler
    ├── URL Frontier (BFS queue, dedup)
    ├── Robots Policy (per-origin caching)
    ├── HTTP Fetcher (httpx)
    ├── HTML Parser (BeautifulSoup)
    ├── DocumentRepository (SQLite)
    └── DocumentIndexer (FieldIndex)
```

## Crawl Lifecycle

1. **Normalize and validate** the seed URL.
2. **Initialize frontier** with the seed at depth 0.
3. **BFS loop** — for each URL in the queue:
   a. Check robots.txt (cached per origin).
   b. Enforce crawl delay between requests.
   c. Fetch the page via HTTP.
   d. Validate content type (HTML only).
   e. Parse HTML — extract title, body text, links, canonical URL.
   f. Store document in SQLite via `DocumentRepository`.
   g. Index title and body via `DocumentIndexer` → `FieldIndex`.
   h. Normalize discovered links and add same-domain ones to the frontier.
4. **Stop** when page limit, depth limit, or empty frontier is reached.
5. **Return** a `CrawlResult` summary.

## URL Normalization

- Resolves relative URLs against the page URL.
- Lowercases scheme and host.
- Removes URL fragments (`#section`).
- Preserves query strings (they may represent distinct resources).
- Rejects unsupported schemes: `javascript:`, `data:`, `file:`, `ftp:`.

## Same-Domain Policy

The crawler only follows links within the same domain as the seed URL.
Domain comparison strips the `www.` prefix, consistent with
`DocumentRepository._extract_domain()`.

| Seed                       | Allowed                            | Blocked              |
|----------------------------|------------------------------------|----------------------|
| `https://example.com`      | `https://example.com/about`        | `https://other.com`  |
| `https://www.example.com`  | `https://example.com/contact`      | `https://google.com` |

## robots.txt Behavior

| Scenario           | Behavior                                         |
|--------------------|--------------------------------------------------|
| 200 response       | Parse rules and enforce                          |
| 404 response       | No restrictions — allow all                      |
| Network error      | Conservative — block crawling for that host       |
| Timeout            | Conservative — block crawling for that host       |
| 5xx response       | Conservative — block crawling for that host       |

Policies are **cached per origin** — robots.txt is fetched only once per
domain per crawl.

## Crawl Limits

| Setting              | Default | Description                            |
|----------------------|---------|----------------------------------------|
| `MAX_CRAWL_PAGES`    | 100     | Max pages successfully stored          |
| `MAX_CRAWL_DEPTH`    | 3       | Max link depth from seed               |
| `CRAWLER_DELAY`      | 1.0     | Seconds between requests               |
| `CRAWLER_TIMEOUT`    | 10      | HTTP request timeout (seconds)         |
| `CRAWLER_MAX_RESPONSE_BYTES` | 5 MB | Max response body size          |

## HTTP Fetching

- Uses `httpx.AsyncClient` with configured timeout and user-agent.
- Follows redirects (up to 10 hops).
- Checks `Content-Length` and actual body size against the limit.
- Only HTML content types are parsed (`text/html`, `application/xhtml+xml`).
- Connection errors, timeouts, and HTTP errors are caught and recorded.

## HTML Parsing

Uses BeautifulSoup with Python's built-in `html.parser`.

| Extracted        | Source                              |
|------------------|-------------------------------------|
| Title            | `<title>` tag                       |
| Body text        | Visible text (stripped of excluded tags) |
| Links            | `<a href=...>` values               |
| Canonical URL    | `<link rel="canonical" href=...>`   |

**Excluded tags** (content removed): `script`, `style`, `noscript`,
`template`, `header`, `footer`, `nav`, `iframe`.

## Document Storage

Uses the existing `DocumentRepository.create_document()`:
- URL uniqueness enforced (duplicates are skipped).
- Domain, content hash, and word count are auto-derived.
- Status is set to `"stored"`.

## Index Integration

Each stored document is immediately indexed via `DocumentIndexer`:
- Title → title index
- Body text → body index
- FieldIndex version remains `2` (no format change).

After crawling, the updated index is persisted to disk.

## Error Handling

- One failed page does not crash the entire crawl.
- Errors are recorded in `CrawlResult.errors` (URLs + messages).
- Stack traces are logged server-side, not exposed via API.

## Security Considerations

- **SSRF prevention**: `localhost`, `127.0.0.1`, `0.0.0.0`, and private
  IP addresses are rejected at the URL normalization layer.
- **Scheme validation**: Only `http://` and `https://` are allowed.
- **Response size limit**: Prevents memory exhaustion from large responses.
- **No script execution**: HTML is parsed as data only.
- **robots.txt**: Respected with conservative failure policy.

## Limitations

- No JavaScript rendering (pages requiring JS will have incomplete content).
- No concurrent fetching (single-threaded per crawl).
- No distributed crawling.
- No sitemap support.
- No PDF/image indexing.
- No recrawling or update detection.
