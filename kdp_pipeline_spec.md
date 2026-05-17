# KDP Coloring Book Pipeline — Production Build Spec

## Context for Claude Code

You are building a production-grade pipeline that generates print-ready KDP coloring books. The user is a senior backend engineer (Go/Fiber + Postgres + Next.js stack). Treat them accordingly — don't over-explain fundamentals, do explain non-obvious choices and tradeoffs.

**Output goal per run:** Given a niche config, produce:
1. Print-ready interior PDF (50 coloring pages, 8.5x8.5", 300 DPI, grayscale, KDP-compliant)
2. Print-ready cover PDF (KDP-spec dimensions calculated from page count)
3. KDP listing metadata (title, subtitle, description, 7 keywords, 2 categories)

**Stack (locked):**
- Python 3.11+
- `uv` for package management
- **Supabase Postgres** (cloud, free tier sufficient)
- `fal-client` → Fal.ai Flux schnell (interior) + Flux dev (cover hero)
- `anthropic` → Claude API for metadata generation
- `psycopg[binary,pool]` (psycopg3, not the older psycopg2)
- `pillow` for image manipulation
- `reportlab` for PDF assembly
- `click` for CLI, `loguru` for logging, `tenacity` for retries, `rich` for terminal UI

---

## Hard requirements — these are non-negotiable

1. **Idempotency.** Every step must be safely re-runnable. If image 23 of 50 fails, re-running picks up from 23, not 1. Use DB state, not filesystem state, as source of truth.
2. **Cost tracking.** Every API call writes a row to `api_calls` with cost. `show-costs <book>` must produce accurate totals.
3. **Atomic state transitions.** Book status moves through a defined state machine. Never write "ready" without all artifacts existing.
4. **No silent failures.** Every exception is caught, logged with context, and either retried or fails the book with a recorded reason.
5. **Reproducibility.** Save the seed and full params for every generated image. If output disappoints, we can re-roll with known parameters.
6. **Manual override at every gate.** Don't auto-proceed from generation to PDF assembly. The user reviews QA results and explicitly approves.

---

## Project Structure

```
kdp_pipeline/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── alembic.ini
├── migrations/
│   └── versions/
│       └── 001_initial.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── settings.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── pool.py
│   │   ├── models.py
│   │   └── repos/
│   │       ├── books.py
│   │       ├── images.py
│   │       └── api_calls.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── fal.py
│   │   └── anthropic.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── images.py
│   │   ├── interior.py
│   │   ├── cover.py
│   │   └── metadata.py
│   ├── qa/
│   │   ├── __init__.py
│   │   ├── image_qa.py
│   │   └── pdf_qa.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── kdp_specs.py
│   │   ├── slugs.py
│   │   └── retries.py
│   └── cli/
│       ├── __init__.py
│       └── commands.py
├── niches/
│   ├── _schema.yaml
│   ├── nurses_v1.yaml
│   ├── cottagecore_mushrooms_v1.yaml
│   └── cozy_seniors_v1.yaml
├── output/                        # gitignored
│   └── {book_slug}/
│       ├── images/
│       │   ├── raw/
│       │   └── filtered/
│       ├── pdf/
│       │   ├── interior.pdf
│       │   └── cover.pdf
│       ├── metadata.json
│       ├── kdp_checklist.md
│       └── run.log
└── tests/
    ├── test_kdp_specs.py
    ├── test_prompt_builder.py
    ├── test_image_qa.py
    └── test_config_loader.py
```

---

## Phase 0: Supabase Setup (user does this before Phase 1)

### User tasks

1. Create Supabase project at supabase.com (free tier is fine)
2. Project Settings → Database → Connection String → **URI**
3. Use the **Session pooler** connection string (port 5432). Do NOT use the transaction pooler (port 6543) — we need prepared statements which transaction pooler doesn't support
4. Save the DB password

### Claude Code tasks

1. Create `.env.example`:
   ```env
   # Supabase (Session pooler, port 5432)
   DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
   
   ANTHROPIC_API_KEY=sk-ant-...
   FAL_KEY=...
   
   OUTPUT_DIR=./output
   LOG_LEVEL=INFO
   LOG_FORMAT=pretty  # 'json' for production
   
   FAL_MAX_CONCURRENT=5
   ANTHROPIC_MAX_CONCURRENT=3
   ```

2. `check-env` command validates connection on demand and prints clear errors for:
   - Wrong port (6543 → suggest 5432)
   - SSL disabled (Supabase requires it; default is `sslmode=require`)
   - Auth failure
   - Missing API keys

3. Verify Supabase specifics:
   - `gen_random_uuid()` works (pgcrypto extension; create if missing in migration)
   - Prepared statements work with the session pooler
   - Pool config: `min_size=1, max_size=10` (well under Supabase free-tier 60-connection cap)

---

## Phase 1: Project Setup

### Tasks

1. `uv init` with Python 3.11 minimum
2. `pyproject.toml` dependencies:
   ```toml
   dependencies = [
     "anthropic>=0.40.0",
     "fal-client>=0.5.0",
     "psycopg[binary,pool]>=3.2.0",
     "pillow>=10.4.0",
     "reportlab>=4.2.0",
     "pypdf>=4.0.0",
     "pydantic>=2.8.0",
     "pydantic-settings>=2.4.0",
     "pyyaml>=6.0",
     "click>=8.1.0",
     "loguru>=0.7.0",
     "tenacity>=9.0.0",
     "rich>=13.7.0",
     "python-dotenv>=1.0.0",
     "alembic>=1.13",
   ]
   
   [dependency-groups]
   dev = [
     "pytest>=8.0",
     "pytest-asyncio>=0.23",
     "pytest-cov>=5.0",
     "mypy>=1.11",
     "ruff>=0.6",
   ]
   ```

3. Configure ruff (lint + format) and mypy (strict on `src/`)
4. `.gitignore`: `.env`, `output/`, `__pycache__`, `.venv`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `assets/fonts/` if licensed

### Acceptance

- `uv sync` succeeds
- `uv run python -m src.main --help` works
- `uv run mypy src/` passes
- `uv run ruff check src/` passes

---

## Phase 2: Database Layer

### Schema (full migration)

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE book_status AS ENUM (
    'created',
    'generating',
    'generation_done',
    'qa_running',
    'qa_done',
    'assembling',
    'metadata_pending',
    'ready',
    'published',
    'failed'
);

CREATE TYPE image_qa_status AS ENUM (
    'pending',
    'passed',
    'rejected_white_pct',
    'rejected_gray_pct',
    'rejected_edges',
    'rejected_margins',
    'rejected_resolution',
    'rejected_manual'
);

CREATE TABLE books (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug            TEXT UNIQUE NOT NULL CHECK (slug ~ '^[a-z0-9_]+$'),
    niche           TEXT NOT NULL,
    status          book_status NOT NULL DEFAULT 'created',
    
    config          JSONB NOT NULL,
    config_hash     TEXT NOT NULL,
    
    title           TEXT,
    subtitle        TEXT,
    description     TEXT,
    keywords        TEXT[] CHECK (array_length(keywords, 1) <= 7),
    categories      TEXT[] CHECK (array_length(categories, 1) <= 2),
    page_count      INT,
    trim_size       TEXT,
    price_usd       DECIMAL(5,2),
    
    asin            TEXT,
    failure_reason  TEXT,
    failure_phase   TEXT,
    
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    generation_started_at   TIMESTAMPTZ,
    generation_finished_at  TIMESTAMPTZ,
    published_at            TIMESTAMPTZ
);

CREATE TABLE images (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id             UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    sequence_num        INT NOT NULL CHECK (sequence_num >= 0),
    
    prompt              TEXT NOT NULL,
    negative_prompt     TEXT,
    seed                BIGINT NOT NULL,
    model               TEXT NOT NULL,
    generation_params   JSONB NOT NULL,
    
    local_path          TEXT,
    fal_url             TEXT,
    file_sha256         TEXT,
    
    qa_status           image_qa_status NOT NULL DEFAULT 'pending',
    qa_metrics          JSONB,
    qa_checked_at       TIMESTAMPTZ,
    
    cost_usd            DECIMAL(8,5) NOT NULL DEFAULT 0,
    
    retry_of_image_id   UUID REFERENCES images(id) ON DELETE SET NULL,
    retry_attempt       INT NOT NULL DEFAULT 0,
    
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE (book_id, sequence_num, retry_attempt)
);

CREATE TABLE api_calls (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id             UUID REFERENCES books(id) ON DELETE SET NULL,
    image_id            UUID REFERENCES images(id) ON DELETE SET NULL,
    
    provider            TEXT NOT NULL,
    endpoint            TEXT NOT NULL,
    operation           TEXT NOT NULL,
    
    request_params      JSONB,
    response_summary    JSONB,
    
    cost_usd            DECIMAL(8,5) NOT NULL DEFAULT 0,
    duration_ms         INT NOT NULL,
    
    success             BOOLEAN NOT NULL,
    error_type          TEXT,
    error_message       TEXT,
    retry_attempt       INT NOT NULL DEFAULT 0,
    
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE book_status_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id         UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    from_status     book_status,
    to_status       book_status NOT NULL,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_books_status ON books(status);
CREATE INDEX idx_books_niche ON books(niche);
CREATE INDEX idx_books_created_at ON books(created_at DESC);
CREATE INDEX idx_images_book_id_seq ON images(book_id, sequence_num);
CREATE INDEX idx_images_qa_pending ON images(qa_status) WHERE qa_status = 'pending';
CREATE INDEX idx_api_calls_book_id ON api_calls(book_id, created_at);
CREATE INDEX idx_api_calls_failed ON api_calls(provider, created_at) WHERE success = false;

-- updated_at trigger
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_updated_at
    BEFORE UPDATE ON books
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Status transition logging
CREATE OR REPLACE FUNCTION log_status_change() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO book_status_log (book_id, from_status, to_status, reason)
        VALUES (NEW.id, OLD.status, NEW.status, NEW.failure_reason);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_status_log
    AFTER UPDATE OF status ON books
    FOR EACH ROW EXECUTE FUNCTION log_status_change();
```

### State machine (enforced in Python)

Legal transitions:
- `created` → `generating`, `failed`
- `generating` → `generation_done`, `failed`
- `generation_done` → `qa_running`, `failed`
- `qa_running` → `qa_done`, `failed`
- `qa_done` → `assembling`, `generating` (for retries), `failed`
- `assembling` → `metadata_pending`, `failed`
- `metadata_pending` → `ready`, `failed`
- `ready` → `published`
- `failed` → (terminal unless manually reset)

`transition_status()` must check legality and raise on violations.

### Migration management

- Use Alembic against `DATABASE_URL`
- First migration creates everything above
- Migrations are reversible (write `downgrade`)

### Connection pool (`src/db/pool.py`)

```python
from psycopg_pool import ConnectionPool
# Singleton pattern
# Use ConnectionPool (sync), not AsyncConnectionPool — we don't need async for DB
# sslmode=require is automatic with Supabase URL
# min_size=1, max_size=10
# Apply pool.open() at app startup; pool.close() on shutdown
```

### Repositories

`src/db/repos/books.py`:
- `create_book(slug, niche, config) -> Book`
- `get_book_by_slug(slug) -> Book | None`
- `get_book_by_id(uuid) -> Book | None`
- `transition_status(book_id, to_status, reason=None) -> None` (with legality check)
- `update_metadata(book_id, **fields) -> None`
- `set_asin(book_id, asin) -> None`
- `fail_book(book_id, phase, reason) -> None`
- `list_books(status=None, niche=None, limit=50, offset=0) -> list[Book]`

`src/db/repos/images.py`:
- `create_image(book_id, sequence_num, prompt, seed, ...) -> Image`
- `update_qa(image_id, qa_status, qa_metrics) -> None`
- `get_passed_images(book_id) -> list[Image]` (ordered by sequence_num)
- `get_failed_images(book_id) -> list[Image]`
- `count_by_status(book_id) -> dict[str, int]`

`src/db/repos/api_calls.py`:
- `log_api_call(book_id, image_id, provider, ...) -> None`
- `total_cost_for_book(book_id) -> Decimal`
- `total_cost_across_books(date_range=None) -> Decimal`

### Rules

- All SQL parameterized — no f-strings in queries, ever
- All writes in transactions (`with conn.transaction():`)
- All repo functions strictly typed
- Models as TypedDict or @dataclass (your call)

### Acceptance

- `uv run alembic upgrade head` runs cleanly against Supabase
- `uv run python -m src.main db-status` shows current revision
- All repo functions pass mypy strict
- Test: create book → transition through statuses → assert log entries

---

## Phase 3: Configuration

### Niche YAML structure

`niches/_schema.yaml` (commented reference):

```yaml
# Slug: lowercase + underscores only. Use _v1, _v2 for versions.
slug: nurses_bold_easy_v1
niche: nurses

book:
  trim_size: "8.5x8.5"
  page_count: 50          # number of designs; total PDF pages = page_count + 2
  price_usd: 9.99
  target_audience: "nurses, BSN students, nursing graduates"

style:
  art_style: >
    bold and easy coloring book, thick black outlines, no shading, no gray,
    pure black line art on white background, large simple shapes
  line_weight: "very thick"
  negative_prompts: >
    shading, gray tones, hatching, color, fill, photorealistic, 3d, depth,
    fine lines, intricate details, text, letters, numbers, signatures,
    watermarks, copyright, borders

subjects:
  - "stethoscope and medical equipment arrangement"
  # ... 25 subjects total

variations_per_subject: 2

composition_modifiers:
  - "centered composition"
  - "with decorative scattered elements around"
  - "with simple geometric border"
  - "on circular background"
  - "diagonal composition"
  - "with small accent shapes in corners"

metadata:
  title_seed: "Nurse Life: A Bold & Easy Coloring Book"
  subtitle_seed: "50 Stress-Relief Designs Celebrating the Nursing Journey"
  keywords_seed:
    - "nurse coloring book"
    - "nursing gifts for women"
    - "RN gifts"
    - "BSN graduation gift"
    - "nurse appreciation"
    - "stress relief coloring book"
    - "nursing student gift"
  categories:
    - "Books > Crafts, Hobbies & Home > Crafts & Hobbies > Drawing"
    - "Books > Health, Fitness & Dieting > Nursing"

cover:
  background_color: "#1a3a52"
  accent_color: "#f4d35e"
  text_color: "#ffffff"
  hero_subject: "stethoscope arranged in heart shape with surrounding medical icons"
  hero_style: "colorful, simple flat illustration, vector style, no text"
  font_family: "Bebas Neue"

generation:
  model: "fal-ai/flux/schnell"
  image_dimensions: [2550, 2550]
  num_inference_steps: 4
  guidance_scale: 0.0
  fixed_seed: null

qa:
  min_white_pct: 90.0
  max_gray_pct: 3.0
  required_white_margin_px: 75
  required_dimensions: [2550, 2550]
  max_retries_per_slot: 3
```

### Pydantic models (`src/config/schema.py`)

Strict validation:
- Slug regex: `^[a-z0-9_]+$`
- Trim size: enum `8.5x8.5` or `8.5x11` (only these for v1)
- Page count: 24–100
- Subjects: non-empty list, len × variations_per_subject must equal page_count
- Hex colors validated via regex `^#[0-9a-fA-F]{6}$`
- Categories: exactly 2 (KDP rule)
- Keywords seed: max 7

### Config loader (`src/config/loader.py`)

- Parse YAML
- Validate with Pydantic
- Compute SHA256 of canonical JSON form for `config_hash`
- Persist snapshot when book created

### CLI

`uv run python -m src.main validate-niche niches/nurses_v1.yaml`

Prints parsed config (or validation errors). Exits non-zero on failure.

### Tests

- Valid configs pass
- Bad slug rejected
- Mismatched page_count/subjects rejected
- Bad hex rejected
- Config hash is deterministic

---

## Phase 4: Provider Layer

### `src/providers/fal.py`

Requirements:
1. Async client; called via `asyncio.run()` from sync code
2. Tenacity retry:
   - Retry on: `RateLimitError`, `httpx.ConnectError`, `httpx.ReadTimeout`, HTTP 5xx
   - Don't retry: 4xx auth, content policy
   - Backoff: 1s, 2s, 4s, 8s, 16s — max 5 attempts
   - Log each retry with reason
3. `asyncio.Semaphore(FAL_MAX_CONCURRENT)` for rate limiting
4. Cost tracking — log to `api_calls` with cost_usd computed from Fal.ai's published pricing. **Confirm exact rates against fal.ai pricing page at build time** (Flux schnell is roughly $0.003 per megapixel; verify before hardcoding).
5. Hard 60s timeout per call
6. Returns: image bytes (downloaded server-side, not URL-only), seed used, duration

### `src/providers/anthropic.py`

Requirements:
1. SDK: `anthropic`
2. Model: `claude-sonnet-4-7` for metadata
3. Same retry policy as Fal
4. Token tracking — log input/output tokens, compute cost
5. JSON-output enforced via system prompt for structured calls

### Both

- All requests/responses logged at DEBUG
- All API calls log to `api_calls` even on failure (cost=0)
- API keys never logged

### Tests

- Mock both providers in unit tests
- One live integration test per provider, marked `@pytest.mark.integration`

---

## Phase 5: Image Generation

### Prompt builder (`src/utils/prompts.py`)

```python
def build_image_prompt(config: NicheConfig, subject: str, variation_idx: int) -> tuple[str, str]:
    """Returns (prompt, negative_prompt). Deterministic given inputs."""
    modifier = config.composition_modifiers[variation_idx % len(config.composition_modifiers)]
    prompt = " ".join([
        config.style.art_style.strip(),
        subject,
        modifier,
        "isolated on pure white background",
        f"{config.style.line_weight} black lines",
        "coloring book page for adults",
        "vector style, professional illustration",
        "clean composition with margin around edges",
    ])
    return prompt, config.style.negative_prompts
```

### Generation flow (`src/generators/images.py`)

1. **Plan stage:**
   - Load book from DB
   - Expand subjects × variations into 50 (sequence_num, subject, variation_idx) tuples
   - Skip any slot with a `qa_status='passed'` image
   - For slots whose latest attempt was rejected and retries remain, queue regeneration
   - For slots over retry limit, mark book failed
   - Print plan: N to generate, estimated cost
   - Confirm with user (unless `--yes`)

2. **Execute stage:**
   - Concurrency: semaphore-bounded
   - Rich progress bar
   - For each result:
     - Save to `output/{slug}/images/raw/{seq:03d}_attempt{retry}.png`
     - Compute file SHA256
     - Insert `images` row with all params
     - Log `api_calls`

3. **Failure handling:**
   - Per-image failures don't kill the batch
   - End-of-batch report: successes/failures
   - Failed prompts dumped to `failed_prompts.txt`

4. **Test mode:**
   - `--test-images 5` generates only 5, exits, no DB record beyond images themselves
   - User reviews quality before committing to full batch

### Acceptance

- Test mode: 5 images, exit cleanly
- Full mode: 50 images, all logged
- Re-running picks up only missing slots
- Total cost: $0.15–0.50

---

## Phase 6: Image QA

### Metrics computed per image (`src/qa/image_qa.py`)

```python
@dataclass
class ImageQAResult:
    white_pct: float           # % pixels with R,G,B all > 240
    near_black_pct: float      # % pixels with R,G,B all < 40
    gray_pct: float            # % pixels with any channel in [50, 200]
    edge_margin_violation: bool # any non-white within required_white_margin_px of edge
    width: int
    height: int
    passed: bool
    reason: str | None         # null if passed
```

Use Pillow + numpy for histogram computations (faster than pure PIL pixel loops).

### Rejection reasons → enum:

- `rejected_white_pct`: white_pct < min_white_pct
- `rejected_gray_pct`: gray_pct > max_gray_pct (shading present)
- `rejected_edges`: content too close to edge
- `rejected_margins`: image doesn't have required clean border
- `rejected_resolution`: wrong dimensions
- `rejected_manual`: human override

### Regeneration loop

After initial pass:
1. Identify rejected slots with retries remaining
2. For each: generate with new random seed, new row in `images` with `retry_of_image_id` set
3. Re-run QA on new images
4. If a slot still rejected after `max_retries_per_slot` → book failed

### Manual review

`uv run python -m src.main review-qa <slug>`:
- Print summary by status
- Generate `output/{slug}/qa_review.html` — grid of all images with status + metrics for easy browsing in browser
- Interactive: `approve <image_id>` / `reject <image_id>` updates DB

### Filtered output

After QA done, copy passed images to `output/{slug}/images/filtered/{seq:03d}.png` in sequence order.

### Tests

- Synthetic all-white image → fails white_pct check (no content)
- Synthetic all-black image → fails (too much ink)
- Synthetic with grayscale gradient → fails gray_pct
- Synthetic clean line art → passes
- Edge-touching content → fails margin

---

## Phase 7: Interior PDF

### KDP specs for 8.5×8.5 paperback coloring book

- Trim: 8.5 × 8.5 inches
- Bleed: 0.125" on outside (top, bottom, outer edge — not inner/binding side)
- Margin: 0.25" safe zone from trim
- Resolution: 300 DPI
- Color: Grayscale
- Format: PDF 1.4+ (PDF/X-1a:2001 preferred)
- All fonts embedded

### Page structure (52 pages total)

- Page 1: Title page — book title + author name
- Page 2: Copyright + instructions:
  ```
  Copyright © {year} {author_name}. All rights reserved.
  
  Tips for coloring:
  • Test markers on the last page first — some may bleed through.
  • Pages are single-sided to prevent bleed-through.
  • Use colored pencils, markers, gel pens, or watercolor pencils.
  • Take your time and enjoy the process.
  ```
- Pages 3–52: 50 designs, one per page, centered

### Implementation notes

```python
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

TRIM = 8.5 * inch
BLEED = 0.125 * inch
SAFE = 0.25 * inch

# For simplicity in v1, use bleed on all 4 sides (slight overprint waste
# but never gets rejected by KDP). Tighter approach (bleed only on outside)
# requires alternating page setup for odd/even — skip that complexity v1.
PAGE_SIZE = (TRIM + 2 * BLEED, TRIM + 2 * BLEED)
```

For each filtered image:
1. Convert to grayscale via Pillow `.convert('L')`
2. Embed at exactly 8.0" × 8.0" centered on page (0.25" safe margin all around)
3. Keep PNG-quality embed (avoid JPEG re-compression)

### PDF QA (`src/qa/pdf_qa.py`)

- Page count = 52
- Every page dimensions correct
- Embedded image DPI ≥ 300
- File size < 150 MB
- PDF opens via `pypdf` without errors
- All fonts embedded (`pypdf.PdfReader` can inspect)

### Acceptance

- `build-interior <slug>` produces `output/{slug}/pdf/interior.pdf`
- All QA checks pass
- Manual visual inspection: open in viewer, all pages render correctly
- File size 30–80 MB typical

---

## Phase 8: Cover

### Cover math (in `src/utils/kdp_specs.py`)

```python
def compute_cover_dimensions(page_count: int, trim_w_in: float, trim_h_in: float, paper: str = 'white') -> dict:
    """
    KDP spine formulas (verify against KDP cover calculator):
    - White paper: 0.002252" × page_count
    - Cream paper: 0.0025" × page_count
    - Color paper: 0.002347" × page_count
    
    Total = back + spine + front + bleed both sides
    """
    spine_factor = {'white': 0.002252, 'cream': 0.0025, 'color': 0.002347}[paper]
    spine = page_count * spine_factor
    bleed = 0.125
    return {
        'total_width_in': (trim_w_in * 2) + spine + (bleed * 2),
        'total_height_in': trim_h_in + (bleed * 2),
        'spine_width_in': spine,
        'bleed_in': bleed,
        'total_width_px': round(((trim_w_in * 2) + spine + (bleed * 2)) * 300),
        'total_height_px': round((trim_h_in + (bleed * 2)) * 300),
    }
```

For 52-page 8.5×8.5 book on white paper: spine ≈ 0.117", total cover 17.367 × 8.75 in = 5210 × 2625 px.

### `src/generators/cover.py`

1. **Generate hero illustration** via Fal.ai Flux **dev** (higher quality, ~$0.025/image):
   - Prompt: `{config.cover.hero_subject}, {config.cover.hero_style}, isolated on transparent or solid color background, no text, no letters`
   - Dimensions: 1500 × 1500
   - Negative: "text, letters, numbers, watermark"

2. **Composite in Pillow:**
   - Canvas at full cover dimensions
   - Fill with `cover.background_color`
   - Place hero on front cover (right 8.5" + half spine area, centered vertically)
   - Render title text (large, top-center of front)
   - Render subtitle below
   - Spine text rotated 90°
   - Back cover: book description + small bullet list of features

3. **Export as PDF** via ReportLab:
   - Embed the composite PNG at full resolution
   - PDF/X-1a:2001 if achievable, else PDF 1.4 with all fonts embedded

### Font handling

- Bundle Bebas Neue (titles) + Open Sans (body) in `assets/fonts/`
- Both SIL Open Font License — commercial use OK
- Add a font loader that verifies they exist on startup

### Variants

Generate 3 layout variants per book:
- A: Hero centered, title above
- B: Hero in lower-right, title fills upper-left
- C: Hero behind a translucent title block

Save as `cover_variant_a.pdf`, `_b.pdf`, `_c.pdf`. User reviews and renames the winner to `cover.pdf`.

### Acceptance

- 3 variants generated
- All 3 match calculated dimensions exactly
- Title readable at 100×150 thumbnail (manual eye test)
- PDFs structurally valid

---

## Phase 9: Metadata

### `src/generators/metadata.py`

Single Claude call with strict JSON output.

System prompt:
```
You generate Amazon KDP book listing metadata. You respond with valid JSON only, no surrounding text, no markdown fences.
```

User prompt template:
```
Generate KDP listing metadata for this coloring book.

CONTEXT:
- Niche: {niche}
- Audience: {target_audience}
- Contents: 50 bold and easy line-art designs themed around {niche}
- Subject areas: {subjects_joined}

CONSTRAINTS:
- title + subtitle combined: max 200 chars
- Title must contain "coloring book" and "{primary_keyword}"
- Description: 1500-2000 chars, 4 paragraphs (hook, what's inside, who it's for, call to action). Plain text, line breaks OK, no emoji, no markdown.
- Keywords: exactly 7, each ≤ 50 chars. Mix broad and long-tail. No brand names, no trademarks.
- Categories: use these exact paths: {categories}

SEEDS (improve them, don't copy verbatim):
- Title: {title_seed}
- Subtitle: {subtitle_seed}
- Keywords: {keywords_seed}

OUTPUT FORMAT (JSON only):
{
  "title": "...",
  "subtitle": "...",
  "description": "...",
  "keywords": ["", "", "", "", "", "", ""],
  "categories": ["", ""]
}
```

Validate response strictly:
- Parses as JSON
- All required fields present
- `title + subtitle` length ≤ 200
- `description` length 1500–2000
- exactly 7 keywords, each ≤ 50 chars, no duplicates
- exactly 2 categories matching config

On validation failure, retry up to 2 times with the validation error appended.

### KDP upload checklist

Auto-generate `output/{slug}/kdp_checklist.md`:

```markdown
# KDP Upload Checklist — {Book Title}

## Files
- Interior PDF: `output/{slug}/pdf/interior.pdf`
- Cover PDF: `output/{slug}/pdf/cover.pdf`

## Listing
- **Title**: {title}
- **Subtitle**: {subtitle}
- **Series**: (leave blank for v1)
- **Author**: {pen_name}
- **Description**: (see description.txt)
- **Keywords** (7 total):
  - {keyword_1}
  - {keyword_2}
  ... 
- **Categories** (2 total):
  - {category_1}
  - {category_2}
- **Age range**: 16+
- **Language**: English
- **Publishing rights**: I own the copyright

## Print settings
- Trim: 8.5 × 8.5 in
- Paper: White
- Ink: Black & white
- Bleed: Yes
- Cover finish: Matte (or Glossy)
- Page count: 52

## Pricing
- Amazon.com: ${price_usd}
- Other marketplaces: auto-convert

## Pre-upload
- [ ] Interior PDF opens cleanly
- [ ] Cover matches calculated dims: {total_w}×{total_h} in
- [ ] All 7 keywords entered
- [ ] Categories selected from KDP browse paths
- [ ] Tax interview (W-8BEN) completed
- [ ] Payoneer USD bank linked

## Post-upload
- [ ] Record ASIN: ____________
- [ ] Run `uv run python -m src.main set-asin {slug} <ASIN>`
- [ ] Wait 72 hours for live status
- [ ] Order author proof copy for QC if first in series
```

### Acceptance

- `generate-metadata <slug>` produces valid `metadata.json`
- Strict validation enforced; bad outputs retried
- Checklist file complete

---

## Phase 10: CLI Orchestration

### Commands

```bash
# Environment
uv run python -m src.main check-env
uv run python -m src.main db-status

# Setup
uv run python -m src.main init-db   # runs alembic upgrade head

# Config
uv run python -m src.main validate-niche <yaml_path>

# Full pipeline
uv run python -m src.main build <yaml_path>                # interactive with gates
uv run python -m src.main build <yaml_path> --yes          # skip prompts
uv run python -m src.main build <yaml_path> --test-images 5  # stops after 5 test images
uv run python -m src.main build <yaml_path> --resume       # continue from current status

# Per-phase
uv run python -m src.main generate-images <slug_or_yaml>
uv run python -m src.main run-qa <slug>
uv run python -m src.main review-qa <slug>
uv run python -m src.main build-interior <slug>
uv run python -m src.main build-cover <slug>
uv run python -m src.main generate-metadata <slug>

# Inspection
uv run python -m src.main list-books [--status STATUS] [--niche NICHE]
uv run python -m src.main show <slug>
uv run python -m src.main show-costs <slug>
uv run python -m src.main show-costs --all [--since YYYY-MM-DD]

# Maintenance
uv run python -m src.main retry-failed <slug>
uv run python -m src.main set-asin <slug> <ASIN>
uv run python -m src.main mark-published <slug>
uv run python -m src.main cleanup-orphans <slug>   # remove unreferenced files
```

### Build orchestration flow

```
build <yaml>:
  1. validate config
  2. create or fetch book record by slug
  3. show cost estimate, confirm (unless --yes)
  4. Phase A: generate images (skipped if already done)
  5. Phase B: run QA
  6. ⛔ if interactive: print summary, exit with message "run review-qa, then build --resume"
  7. Phase C: assemble interior PDF
  8. Phase D: generate cover (3 variants)
  9. ⛔ if interactive: exit with "pick a variant, rename to cover.pdf, then build --resume"
  10. Phase E: generate metadata
  11. Phase F: status → ready, print final summary with all paths and costs
```

### Final summary print:

```
✓ Book ready: nurses_bold_easy_v1

  Title: Nurse Life: A Bold & Easy Coloring Book
  Pages: 52
  Trim: 8.5 × 8.5 in
  Price: $9.99

  Artifacts:
    Interior: output/nurses_bold_easy_v1/pdf/interior.pdf
    Cover:    output/nurses_bold_easy_v1/pdf/cover.pdf
    Metadata: output/nurses_bold_easy_v1/metadata.json
    Checklist: output/nurses_bold_easy_v1/kdp_checklist.md

  Cost breakdown:
    Fal.ai:     $0.42 (54 images including retries)
    Anthropic:  $0.03 (1 metadata call)
    Total:      $0.45

  Next: follow output/nurses_bold_easy_v1/kdp_checklist.md
```

### Acceptance

- End-to-end build produces all artifacts
- Total runtime: < 20 minutes
- Total cost: < $0.75
- Re-running after interruption resumes correctly
- All state changes logged

---

## Testing requirements

Unit tests in `tests/`. Use pytest.

**Required:**
1. `test_kdp_specs.py`: cover dimension math against known page counts
2. `test_config_loader.py`: valid configs, invalid slugs, mismatched page counts, bad hex
3. `test_prompt_builder.py`: deterministic output for same inputs
4. `test_image_qa.py`: synthetic test images (all-white, all-black, with-shading, clean-line-art) → correct verdicts
5. `test_db_repos.py`: against local Postgres in CI (or skipped if no DATABASE_URL_TEST)

Mock providers in unit tests. Integration tests in `tests/integration/` marked `@pytest.mark.integration`, skipped by default.

### Acceptance

- `uv run pytest` passes all unit tests
- `uv run pytest --cov=src` shows ≥ 70% coverage on `src/`

---

## Operational concerns

### Logging

- `loguru`: JSON in production, pretty in dev (via `LOG_FORMAT`)
- Per-book log file at `output/<slug>/run.log` in addition to stdout
- Every status transition, every API call, every QA verdict logged

### Error recovery

- All operations resumable from DB state
- Process killed mid-generation → re-run picks up exactly where it stopped
- `cleanup-orphans <slug>` removes filesystem files not referenced in DB

### Cost monitoring

- `show-costs <slug>` per-book breakdown
- `show-costs --all` total spend (optional date range)
- Optional `MAX_BOOK_COST_USD` env var — abort generation if projected cost exceeds

### Secrets

- All keys in `.env`, never committed
- `.env.example` is source of truth
- Startup fails loudly if any required env var missing

---

## Sequencing for Claude Code

**Do not build all phases at once.** Build in this order, stopping at each gate:

1. **Phase 0**: User sets up Supabase
2. **Phase 1**: Project skeleton → run `--help`
3. **Phase 2**: DB layer + first migration → run against Supabase, verify tables
4. **Phase 3**: Config layer → load nurses_v1.yaml, print parsed
5. **Phase 4**: Provider layer → one test call to each API
6. ⛔ **GATE: Phase 5 small batch** — generate 5 test images, save locally, **STOP and show user**. Pipeline quality lives or dies here. Iterate prompts together if quality is off.
7. **Phase 5 full**: scale to 50 once 5 look right
8. **Phase 6**: QA pipeline → run on 50, show breakdown
9. ⛔ **GATE: Manual QA review** — user inspects filtered/
10. **Phase 7**: Interior PDF → open and inspect
11. **Phase 8**: Cover → 3 variants, user picks
12. **Phase 9**: Metadata → user reviews JSON
13. **Phase 10**: Wire up `build` command
14. Run end-to-end on a second niche (cottagecore) to validate generalization

### The Phase 5 gate is critical

If Fal.ai images come out with gray patches, missing detail, or wrong style, no QA filter or PDF assembly fixes it. Tune the prompts at that gate. Iterate until 4 of 5 test images are publication-grade before scaling to 50.

---

## Questions to confirm with user before starting Phase 1

1. Supabase project created? Session pooler connection string in hand?
2. Fal.ai account active? `FAL_KEY` ready?
3. Anthropic API key? (Same key used elsewhere is fine.)
4. Pen name for KDP author field? (Required on cover and metadata.)
5. Confirm first build target: `nurses_bold_easy_v1`?
6. Log format preference during dev: `pretty` or `json`?

---

## What success looks like

- One CLI command produces a publishable book
- Every artifact reproducible from DB state
- Every dollar of API spend tracked
- No silent failures
- Book 1: ~20 minutes runtime, < $0.75 in API costs
- Books 2-N benefit from prompt learnings
- Architecture extensible to Etsy printables (Phase 11) and eventually shorts-as-a-service (different content, same pattern)
