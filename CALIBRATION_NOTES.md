# Calibration notes

Why the pipeline's tunable thresholds are set where they are. Read this before
changing any of them — most were picked from measured data, not guessed.

## Vision QA pass threshold — `DEFAULT_PASS_THRESHOLD = 80`

`src/qa/vision_qa.py`. A page passes Vision QA when its rubric score is `>= 80`
and it raises no red flag.

### How 80 was chosen

Calibrated on the 5 Q2 test images (`output/nurses_bold_easy_v1/test_q2_lora/
verify_t200/`) — a deliberate mix of strong and weak pages. Each was graded by
`claude-sonnet-4-6`; the teddy and heart, which sat near the boundary, were
graded **7 times each** to measure run-to-run variance.

| Image | Score (range over runs) | Human verdict |
|-------|-------------------------|---------------|
| nurse + patient | 84–85 | good — should pass |
| kawaii heart | 83–87 | good — should pass |
| teddy bear (messy body, weak face) | 66–74 | bad — should reject |
| IV bag (tiny on page, ambiguous) | 66 | bad — should reject |
| cat instead of stethoscope | 62–67 | bad — should reject |

The scores fall into two clusters with a **clean 9-point gap**:

```
reject cluster ........ 62 ──────── 74      (gap)      83 ──────── 87 ........ pass cluster
                                          75   80
```

The threshold belongs in the middle of the gap, not at its edge:

- **75** (the spec's starting value) sits at the bottom edge. The teddy scored
  74 in 2 of 7 runs — a 1-point margin. A noisier run scoring it 75–76 would
  false-pass a genuinely bad page.
- **80** sits mid-gap: the teddy clears rejection by ~6 points, the genuine
  passes (83+) clear by ~3–7, and nothing observed lands in 75–82.

Verified at 80: nurse 85 ✓ pass, heart 87 ✓ pass, teddy 74 ✓ reject, IV bag 66
✓ reject, cat 67 ✓ reject.

### Why not fix it with the rubric instead

The teddy is a death-by-a-thousand-cuts image (messy body, blob feet, weak
face) — not a single dramatic defect. Tuning the face red-flag wording to fail
it is a proven seesaw: wording strict enough to red-flag the teddy's so-so face
also red-flags the kawaii heart's so-so face (a false positive). The teddy
correctly fails on *aggregate score*; it just needs threshold headroom. Hence
the threshold move rather than more rubric wording.

### Per-niche caveat

80 was calibrated on the **nurses** niche. A new niche may cluster differently —
re-probe its test images before trusting 80 there, and adjust with
`tune-vision-threshold <slug> <n>` (re-applies pass/fail to recorded scores, no
regeneration).

### Rubric prompt changes made during calibration

`USER_PROMPT_TEMPLATE` / `SYSTEM_PROMPT` in `src/qa/vision_qa.py`:

1. **No niche-fit penalty.** Subjects are pre-curated in the niche YAML; the
   rubric judges only how well the requested subject is *rendered*, never
   whether it "fits" the niche. (Without this, a plain heart lost ~8 points for
   not being nurse-themed.)
2. **Face/eye distortion red flag — softened.** Catches *clearly broken* faces
   (structural melts, fused eyes, mismatched character type) but explicitly not
   deliberate kawaii minor asymmetry. The first draft ("even subtle
   deformations") false-positived the clean kawaii heart.
3. **Bold & Easy simplicity is a feature.** The rubric must not deduct for "too
   simple for adults" — simplicity is the intended style.

Also: a "subject fundamentally different" red flag now maps to
`rejected_vision_subject` (was defaulting to `rejected_vision_composition`).

## Line-art ink threshold — adaptive, since Q4 (`src/utils/line_art.py`)

`normalize_line_art` binarises each generated page to pure black-on-white. The
cutoff that decides "ink vs background" has been retuned twice.

### The fixed cutoff and why it failed twice

- **128 → 200 (Q2).** The coloring-book LoRA renders complex scenes in pale
  grey (drawn-pixel brightness up to ~240); the original midpoint cutoff of 128
  silently dropped those strokes. 200 recovered most of them.
- **200 still dropped pages (Q4).** The teddy-bear raw (`subj19`) came back with
  a body drawn at brightness **~244** — a clean, complete bear, but paler than
  the fixed cutoff. Binarising at 200 erased the body; Vision QA scored the
  wreckage **22/100**. The LoRA's line darkness varies run to run, so *no* fixed
  cutoff is safe: set it high enough to catch a pale page and it hardens
  background noise on a dark one.

### The adaptive threshold

The cutoff is now derived per image (`_adaptive_threshold`):

- background = mean brightness of four 100×100 **corner** patches (`_CORNER_PATCH`).
  Corners, not a whole-image statistic — a pale interior fill would skew the
  whole-image mean toward the ink.
- threshold = `round(background) - 7` (`_BACKGROUND_MARGIN`). Ink is anything
  meaningfully darker than the page's own paper.
- floored at **180** (`_THRESHOLD_FLOOR`): a page whose corners are themselves
  dark is a degenerate near-black generation — flooring keeps binarisation
  bounded so pixel QA can still *reject* it rather than the threshold collapsing
  to recover noise.

The per-image threshold is recorded in `generation_params.line_art_threshold`
for audit. (It applies only to the binarising post-process modes — see below.)

**Measured on the Q4 re-gate:** re-processing the teddy raw with the adaptive
threshold scored it **70/100** (was 22) — the bear is now fully recovered and
gradeable. Other raws were unaffected (their backgrounds were already near-white,
so the adaptive cutoff landed close to 200 anyway).

`Otsu` was tried and rejected: on the teddy it picks ~130, latching onto the
high-contrast black face and re-dropping the pale body. Background-relative is
the correct model — the question is "darker than *this page's* paper", not
"which global cluster".

- `qa.min_white_pct = 85` in `niches/nurses_v1.yaml` (was 90). Bold & Easy line
  art legitimately carries more ink than thin line art.

## Line-art post-process — three modes (`src/utils/line_art.py`)

**The post-process must match the LoRA.** `normalize_line_art` was built for
the pale, thin strokes of earlier phases — it binarised every page and applied
a fixed `MinFilter` dilation. But the coloring-book LoRA on the cottagecore
niche produces publication-ready line art on its own, and that pipeline only
degrades it: binarising jags the smooth antialiased edges, and on a detail-rich
page dilation merges closely-spaced strokes (foliage, thatch, flower clusters)
into solid black. The cottagecore wildflower bouquet went from a clean ~9%-ink
raw to a ~45%-ink silhouette — the post-process, not the model, was the loss.
Reviewing all five cottagecore calibration subjects, the raw beat the processed
version on every one.

The post-process is therefore niche-selectable via `post_process.mode`:

| mode | what it does | for |
|---|---|---|
| `minimal` | upscale + whiten the background only | a LoRA that already draws clean line art (cottagecore) |
| `dilate`  | binarise + bold every stroke, fixed `MinFilter(5)` | a LoRA whose output is uniformly pale and thin |
| `auto`    | binarise + bold, dilation window per image by ink density | mixed output that sometimes needs bolding (nurses) |

`minimal` is the default, and the canonical pipeline output for cottagecore:
the raw, upscaled to print resolution with the background clipped to pure
white, nothing else — no binarise, no dilation.

### When dilation helps, and when it hurts

- **Helps** — pale, sparse output (the nurses niche). Thin strokes genuinely
  need bolding, and the pages are sparse (Q2/Q4 raws all 0–11% ink) so dilation
  has no closely-spaced strokes to merge.
- **Hurts** — clean, detail-rich output (the cottagecore niche). The strokes
  are already bold enough; binarising jags them and dilation merges the detail.

### The `auto` density buckets

When `auto` does dilate, the `MinFilter` window is chosen per image from the
binarised page's ink fraction (`_dilation_window`):

| ink fraction | window | rationale |
|---|---|---|
| < 7%   | MinFilter(5) | sparse — strokes have room; full bolding is safe |
| 7–9%   | MinFilter(3) | moderately dense — a gentle 1-px bold |
| ≥ 9%   | none         | dense — dilation would merge strokes into masses |

Cutoffs read off 36 measured nurses + cottagecore raws: nurses' sparse subjects
sit below 7% and take the full window safely; cottagecore's foliage/thatch
pages sit at 9–15% and overshoot the QA white-page floor (`min_white_pct = 85`
→ ≤ 15% ink) if dilated.

### Why density, not darkness

A darkness signal — skip dilation when the ink is *already dark* — was proposed
and measured-and-rejected: median ink brightness is dominated by the
antialiasing skirt and swings run-to-run on the *same subject* (the teddy-bear
raws measured `background − median` at 253, 11, 20 across three attempts; the
stethoscope at 239, 191, 149, 14). Ink fraction is stable and directly predicts
the over-ink failure.

### Evidence

A direct A/B: the cottagecore bouquet raw, `minimal`-processed, scored Vision QA
**87/100 PASS**; the over-dilated version of the same generation scored
**70/100 REJECT** (`rejected_white_pct`) — the post-process alone cost 17 points
and a QA failure. The nurses Q2/Q4 raws were checked (0–11% ink): dilation never
merged or destroyed them, so **those gate results stand, uncorrupted**.

`normalize_line_art` returns a `LineArtResult`; `generation_params` records
`line_art_mode`, `line_art_threshold`, and `line_art_dilate_window` per image.

## Subject swap — `subj08` (Q4)

`niches/nurses_v1.yaml` subject 08 has been changed twice:

1. **"a cute blood pressure cuff with a heart design"** — the LoRA rendered it
   as an ambiguous heart-blob; Vision QA could not read it as a BP cuff.
2. → **"medical scrubs hanging on a hook"** — the LoRA drew *two* outfits on
   clothes hangers (a duplicate-subject failure), scored 72.
3. → **"a single scrub top hanging on a wall hook"** — wording made explicitly
   singular. If retries still duplicate it, the structural fix is to add
   "a single instance only, no duplicates or multiples of the subject" to the
   `object` `kind` directive in `src/utils/prompts.py`.

## Subject scale — unsolved by prompt wording (Q4 → Q5)

Q4 added a prompt clause asking for the subject "filling about 70% of the page".
**Two independent generations proved Flux+LoRA does not honour framing
language** — the nurse scene and the syringe both came back at ~20–40% of the
page regardless. The clause was **removed** in the Q4 prompt rework; it was
noise, not signal.

This is not fixed — it is deferred:

- **Q5 (style reference)** is the real fix. Once image-to-image / a reference
  image is in play, the reference *is* the framing constraint — the model
  matches the reference's subject scale instead of being asked in words.
- **Q6 (tooling)** should add a deterministic fallback: a "post-process
  auto-crop and recenter" step in `normalize_line_art`. Detecting the ink
  bounding box and rescaling it to a target fill fraction is cheap, deterministic,
  and independent of the model — a safety net for when a generation still comes
  back small. Worth implementing alongside the adaptive threshold.

## Future tooling

The 7-runs-per-image variance probe used to calibrate the Vision QA threshold
proved its worth — single-shot scores hid that the teddy straddled the
boundary. A `probe-vision-qa <image> --runs N` CLI command (score distribution
+ red-flag frequency for one image) is planned for Q6; useful for calibrating
new niches, debugging an unexpected reject, and telling rubric signal from
run-to-run noise.
