# Laboratory evidence audit — 2026-11-12

Technical deliverable checklist against [the original assignment](assignment/README.md).
This is not the student report. Repository files and previously recorded local
measurements were inspected. Current laptop execution needs a fresh local
`scripts/check_lab.py` result; Git commits are not live process observations.

| Requirement | Evidence present | Remaining / not established |
| --- | --- | --- |
| Data preparation | Conversion, ignored/non-target handling, EDA and GT visualization code; local notes document 24,201 train, 2,846 val and 3,811 test frames | Inspect generated EDA plots/GT boxes and connect observations to experiment choices |
| Two-stage detector | Faster R-CNN trained in recorded logs; configs for proposals/RoI variants; observed public result 0.1851681872 | Controlled results for at least two relevant parameter experiments, comparable speed/size metrics and three explained failure cases are not established by config files |
| One-stage anchor-based | RetinaNet trained; observed public 0.1791370309; small-anchor/GIoU experiment code/configs | Measured baseline-vs-tuned comparisons, data-backed anchors and required loss/tuning evidence need artifact verification |
| One-stage anchor-free | FCOS training and validation documented; assignment/loss variants configured | Completed controlled tuning results for at least two parameters and comparison by object size need artifact verification; no FCOS public score is recorded |
| Query-based | RT-DETR trained; LR, tiling, crop/full-frame and interpolation experiments recorded; public best 0.2021478454 | Same-budget curve against a CNN, query/matching/loss parameter results and small/dense-scene analysis are not all established |
| Own pipeline | Warm starts, measured full-train coverage, native/tiled export and negative experiments recorded | A successful generalizing improvement and final reproducible recipe still need selection; crop and later full-train public regressions must remain in experiment history |
| CV / OOF | Fold generation, family-isolation checks and OOF aggregation code; three new family folds prepared | No completed honest three-fold/OOF campaign is evidenced. Original clip-level folds contain related-flight leakage. c1 cannot initialize a fold whose held-out families it already trained on |
| Metrics and speed | AP/AP50/AP_small, per-class metrics and inference timing are implemented and occur in some logs | A verified comparison table for all four families on identical data, settings and compute budget is not established |
| All labeled data | Two 24,201-frame full-train passes verified for c1 continuation | Final train+val (27,047 labeled frames in the recorded manifest) was held behind STOP; completion is not established. Train loss/seen-val AP cannot replace independent selection evidence |
| Competition baseline | Valid submissions and precise recorded best public 0.2021478454 | Baseline 0.2216324580 is not exceeded; gap 0.0194846126. Target 0.28+ is a separate goal, not a measured score |
| Student report and submission | Reproducible CLI/docs and archived assignment exist | Course-template report, GitHub Pages publication, LMS submission, final two choices and defense readiness are not verified. The assignment prohibits an AI-written report |
| Bonuses | Diagnostic tooling exists | Temporal prediction use, feature/attention/Grad-CAM evidence and W&B artifacts are not established. No W&B integration was found in the current code |

## What is happening versus what is planned

- The repository now contains stronger candidate gates and shared evaluation
  receipts. A commit does not establish that the laptop has pulled or activated
  them. A healthy current job should reach a safe saved boundary before updating.
- The latest reviewed local note described alpha=0.5 direction validation as a
  next/pending experiment. This audit does not label it currently running or done.
- Original c1 public score is the preserved best. The 224 continuation improved
  local AP but has a recorded approximate public score 0.1919 and must not be
  an automatic training source. The selection signal needs correction.
- Configurations and passing unit tests prove available code, not performed
  experiments, trained fold models, OOF predictions or academic completion.

## Order of work

1. Apply the source-bound robust selection protocol at the next safe boundary;
   preserve c1, verify actual Kaggle metric details, reuse evaluations and record
   improvement per compute time. Do not spend more hours following a rejected
   direction based only on pooled AP.
2. Run the local artifact auditor and list actual missing results for the four
   model rubrics. Generate plots/EDA/failure-example artifacts on CPU while GPU
   work proceeds. Students write their own interpretations and report.
3. Budget an honest family-grouped CV/OOF path with initialization compatible
   with each held-out fold; do not relabel c1 warm starts as clean OOF.
4. Select a fixed final recipe, then fit all labeled train+val if appropriate,
   finish/validate export and make the required final selection/submission.
   Verify the latest course cutoff before commits/LMS delivery.

Local command (CPU-only, no tensor unpickling):

```bash
python3 scripts/check_lab.py --work "$CVLAB_LOCAL_WORK" \
  --out "$CVLAB_LOCAL_WORK/checks/status.json"
```

The JSON includes local run metrics, historical versus independently eligible
results, inherited family-leakage flags, active process argv and recorded plan
states. It deliberately leaves CV/OOF and grading readiness unverified where
evidence is insufficient. Do not publish its raw machine/runtime metadata.
