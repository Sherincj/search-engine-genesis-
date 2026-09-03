# Text Processing — Technical Documentation

## Overview

The text-processing module (`app/text_processing/`) provides the tokenisation
pipeline for the search engine. It converts raw document text into a
deterministic list of normalised tokens.

This module is the **shared foundation** for all future search components:

```
                      Documents    Queries
                          │           │
                          └─────┬─────┘
                                │
                                ▼
                           TOKENIZER
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
                Tokens      Word Count   TF Data
                    │
                    ▼
              Inverted Index  (Phase 4)
                    │
                    ▼
                 TF-IDF       (Phase 6)
                    │
                    ▼
                  BM25         (Phase 7)
                    │
                    ▼
             Search Results
```

Both **documents** and **search queries** go through the same tokeniser with
the same normalisation rules. This is critical — if documents are tokenised
one way and queries another, search quality will suffer.

---

## Pipeline

```
Raw text
   ↓
1. Unicode normalisation (NFC)
   ↓
2. Case normalisation (lowercase)
   ↓
3. Apostrophe stripping
   ↓
4. Token boundary detection (regex: Unicode letters + digits)
   ↓
5. Optional stop-word filtering
   ↓
Tokens  →  list[str]
```

Each stage is independently testable and replaceable.

---

## Module Structure

```
app/text_processing/
├── __init__.py        # Public API exports
├── types.py           # TokenizerConfig dataclass
├── normalizer.py      # Unicode + case + whitespace normalisation
├── stopwords.py       # Stop-word list and filter function
└── tokenizer.py       # Tokenisation pipeline + utilities
```

---

## Normalisation (`normalizer.py`)

`normalize_text(text: str) -> str`

1. **Unicode NFC** — compose combining characters (e.g. `e` + `◌́` → `é`).
2. **Lowercase** — `Python` → `python`.
3. **Whitespace collapse** — multiple spaces / tabs / newlines → single space;
   leading and trailing whitespace stripped.

```python
>>> normalize_text("  Python   IS\nGREAT  ")
'python is great'
```

---

## Tokenisation Rules (V1)

### Hyphens → split

```
web-development  →  ["web", "development"]
state-of-the-art →  ["state", "of", "the", "art"]
```

Hyphens act as token boundaries. No linguistic interpretation.

### Apostrophes → strip

```
don't   →  ["dont"]
user's  →  ["users"]
James'  →  ["james"]
```

ASCII `'` (U+0027), left `'` (U+2018), and right `'` (U+2019) are all
stripped before tokenisation. This is consistent and deterministic.

### Numbers → preserved

```
2026     →  ["2026"]
RTX 5060 →  ["rtx", "5060"]
```

Numbers are important for search (`"python 3.12"`, `"iphone 17"`).

### Decimals → split on dot

```
3.14       →  ["3", "14"]
Python 3.12 →  ["python", "3", "12"]
```

The dot is not part of the token pattern. Queries go through the same
pipeline, so `3.12` in a query also becomes `["3", "12"]` — matching works.

### Alphanumeric → kept whole

```
python3   →  ["python3"]
web3      →  ["web3"]
rtx5060   →  ["rtx5060"]
windows11 →  ["windows11"]
```

Mixed letter-digit sequences are preserved as single tokens.

### Underscores → split

```
my_function →  ["my", "function"]
```

Underscores act as token boundaries.

### Unicode → supported

```
café      →  ["café"]
naïve     →  ["naïve"]
résumé    →  ["résumé"]
中文       →  ["中文"]
日本語     →  ["日本語"]
한국어     →  ["한국어"]
```

The regex `[^\W_]+` with `re.UNICODE` matches sequences of Unicode
letters and digits from any script.

### Duplicate tokens → preserved

```
python python python web  →  ["python", "python", "python", "web"]
```

**Duplicates are never removed.** Term frequency depends on them.

---

## Configuration

```python
from app.text_processing import Tokenizer, TokenizerConfig

# Default: lowercase on, Unicode on, stopwords off
t = Tokenizer()
t.tokenize("Python is GREAT!")  # ['python', 'is', 'great']

# Custom config
t2 = Tokenizer(TokenizerConfig(
    lowercase=True,
    normalize_unicode=True,
    remove_stopwords=True,
))
t2.tokenize("Python is a language")  # ['python', 'language']

# Disable lowercase
t3 = Tokenizer(TokenizerConfig(lowercase=False))
t3.tokenize("Hello World")  # ['Hello', 'World']
```

### Default configuration

| Setting             | Default | Reason                                    |
|---------------------|---------|-------------------------------------------|
| `lowercase`         | `True`  | Case-insensitive matching                 |
| `normalize_unicode` | `True`  | Consistent handling of combining characters|
| `remove_stopwords`  | `False` | Preserve all tokens by default            |

---

## Stop Words

Stop-word removal is **disabled by default**.

Reason: some queries depend on stop words for meaning:

```
"to be or not to be"
"The Who"
```

When stop-word removal is enabled, a conservative English list (~55 words) is used:

```python
from app.text_processing import tokenize

tokenize("python is a language")
# ['python', 'is', 'a', 'language']

tokenize("python is a language", remove_stopwords=True)
# ['python', 'language']
```

Custom stop-word sets can be passed to `filter_stopwords()` directly.

---

## Term Frequency

A utility for counting token occurrences:

```python
from app.text_processing import tokenize, term_frequency

tokens = tokenize("Python Python web Python web")
freq = term_frequency(tokens)
# {'python': 3, 'web': 2}
```

This will later feed TF-IDF and BM25.

---

## Word Count

Documents use `token_count()` (equivalent to `len(tokenize(text))`) for the
`word_count` field, replacing the Phase 2 whitespace-based `text.split()`.

This matters for text with punctuation or hyphens:

| Text              | `text.split()` | `tokenize()`          | Count |
|-------------------|---------------|-----------------------|-------|
| `hello,world`     | 1             | `["hello", "world"]`  | 2     |
| `web-development` | 1             | `["web", "development"]` | 2  |
| `don't stop`      | 2             | `["dont", "stop"]`    | 2     |

> **Note:** Documents created under Phase 2 may have different `word_count`
> values than documents created under Phase 3+ if their text contained
> punctuation without spaces. This is expected in development.

---

## Stemming

Stemming and lemmatisation are **not implemented** in V1.

```
programming  →  programming   (NOT "program")
running      →  running       (NOT "run")
cars         →  cars          (NOT "car")
```

The search engine preserves original normalised word forms. Stemming can
be evaluated later based on actual search-quality requirements.

---

## API Reference

### Convenience functions

```python
tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]
normalize_text(text: str) -> str
term_frequency(tokens: list[str]) -> dict[str, int]
token_count(text: str) -> int
filter_stopwords(tokens: list[str], stop_words=None) -> list[str]
```

### Class-based

```python
Tokenizer(config: TokenizerConfig | None = None)
    .tokenize(text: str) -> list[str]

TokenizerConfig(
    lowercase: bool = True,
    normalize_unicode: bool = True,
    remove_stopwords: bool = False,
)
```

### Error handling

- Passing `None` or a non-string to `tokenize()` raises `TypeError`.
- Empty or whitespace-only input returns `[]` (no exception).
