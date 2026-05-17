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

## Pixel QA / line-art thresholds (Q2)

For completeness — calibrated earlier, separately from Vision QA:

- `_INK_THRESHOLD` / `_UPSCALE_CUTOFF = 200` (`src/utils/line_art.py`). The
  coloring-book LoRA renders complex scenes in pale grey (drawn-pixel brightness
  up to ~240); the original midpoint cutoff of 128 silently dropped those
  strokes. 200 recovers them without hardening background noise.
- `qa.min_white_pct = 85` in `niches/nurses_v1.yaml` (was 90). Bold & Easy line
  art legitimately carries more ink than thin line art.

## Future tooling

The 7-runs-per-image variance probe used to calibrate the Vision QA threshold
proved its worth — single-shot scores hid that the teddy straddled the
boundary. A `probe-vision-qa <image> --runs N` CLI command (score distribution
+ red-flag frequency for one image) is planned for Q6; useful for calibrating
new niches, debugging an unexpected reject, and telling rubric signal from
run-to-run noise.
