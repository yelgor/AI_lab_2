# Validation quality and isolation

Verified on 2026-11-11 against the current RT-DETR source and trial split files
and the official validation manifest. Machine-readable evidence is in
`local_supervision/validation_isolation.json` (local, gitignored).

| Partition | Frames | Flight families | Purpose |
| --- | ---: | ---: | --- |
| Current training fold | 16121 | 32 | Gradient updates |
| Fixed search probe | 512 | 16 | Repeated candidate selection |
| Original fold validation | 8080 | 19 | Historical comparison; includes related flights |
| Official validation | 2846 | 7 | Full confirmation; repeatedly used for selection |
| Competition test | 3811 | Unknown here | External scored evaluation; labels unavailable locally |

The search probe and official validation have no training flight-family overlap.
They also have no flight-family overlap with each other. Random frame splitting
would not provide this protection for temporally correlated video data.

The original 8080-frame fold validation overlaps training in flight families
`uav0000013`, `uav0000072`, and `uav0000266`. Its full native AP is a historical
auxiliary comparison, not an independent generalization estimate. The current
already-running legacy process uses this historical comparison as one promotion
gate. New invocations use `unseen-flight-families-v2`: full native confirmation
excludes the union of candidate/champion/original training families, including
recorded inherited training exposure, and requires at least four remaining
families. Both models use identical records and settings. Record fingerprints
prevent reuse of incompatible score caches. Previous confirmation decisions
without this protocol tag do not satisfy the new confirmation gate. This does
not retroactively change the active process or its recorded metrics.

The probe and official validation have already influenced model/configuration
selection. Neither can be described as an untouched final test. Creating a new
subset from already consulted validation data does not erase that history.
An additional final estimate needs genuinely unconsulted labeled flights or
the competition's external evaluation. Do not train on official validation to
improve its reported score. Synthetic augmentations remain training data, not
independent evidence of generalization.

Accept quality improvements using measured full-data AP, per-class AP, small-object
AP and paired flight-level behavior, with qualitative inspection of missed
objects, false positives and crowded scenes. Training loss and a structurally
valid CSV alone do not establish prediction quality. Report local selection
results separately from observed competition scores. Keep the original public
best checkpoint and CSV until an improvement has actually been established.

Further cross-validation must group entire flight families, including excerpts,
and initialize each fold from weights that have not been fine-tuned on that
fold's held-out families. Warm-starting every fold from the current detector
would invalidate that independence. Do not launch such training concurrently
with the healthy current GPU job.

## Family-grouped experiments and stopping

`scripts/prepare_family_folds.py` writes separate, versioned flight-family folds
without touching the legacy `folds.json`. The prepared local
`folds_family_v1.json` has three validation partitions: 8079, 8029, and 8093
frames, with 16 independent flight families each. Every training-manifest
frame belongs to exactly one validation partition. The local table is
`local_supervision/family_folds.csv`.

`configs/rtdetr_family_generalization.yaml` selects this file through
`data.folds_file`, enables the randomized flight/time sampler, preserves weight
decay and random horizontal flips, freezes BatchNorm moments, and sets validation
early stopping to three checks without an AP improvement greater than 0.001.
Training uses blocks of at most 2000 batch-one frames with a fixed, temporally
spread 512-frame unseen-family probe between blocks, so stopping can react
within the wall budget. These blocks are partial epochs, not complete dataset
passes; different blocks may revisit frames. Full held-out evaluation requires
`evaluation.validation_images=null` after selecting weights; probe AP alone is
not the final quality result. The official validation stays outside training.
These are explicit starting heuristics, not proof of generalization. Best weights
are selected by validation AP; last weights remain available for resume. The
bounded LR race retains its own fixed-budget/pruning protocol rather than
silently changing a running comparison. Random crops are a separate ablation;
mixing them into the sampler comparison would confound its result.

The engine now rejects same-flight validation exposure both from the current
training records and from inherited fine-tuned weights. New run split metadata
records this inherited exposure. Never relabel old fine-tuned weights as a clean
cross-validation baseline. The new folds/config are ready, but a fresh full
cross-validation campaign has not been run. The original laboratory assignment was subsequently fetched from the competition
PageService API on 2026-11-11 and cached locally in
`local_supervision/kaggle_ListPages.json` (Description). These safeguards alone
do not establish that all grading requirements are complete.

## Randomized sampling ablation

`loader.sampling=flight_time` interleaves randomized flight-family queues and eight chronological bins per family, with each eligible frame visited once per epoch. Short prefixes favor family/time coverage and therefore differ from uniform frame sampling; improvement is not assumed. Epoch-boundary resume uses the checkpointed loader RNG. Earlier partial epochs can overlap later epochs. Existing shuffle remains the default and validation order is unchanged.

`scripts/start_sampling_experiment.py --work WORK --frames 72000 --hours 8
--reserve-hours 2` queues a larger continuation behind the shared GPU lease.
It warm-starts the original learned weights AND optimizer/RNG, with the same LR
and fixed independent 512-frame probe as cycle_001/trial_0. Output:
`WORK/runs/rtdetr_sampling_flight_time`. Coverage artifacts count actual input
frames before augmentation.

The first 6000-frame block remains a controlled sampling comparison. Subsequent
blocks visit the complete eligible 16121-frame training pool in randomized
flight/time order, capped at 72000 presentations total. The planned blocks are
6000 + 16121 + 16121 + 16121 + 16121 + 1516; these are repeated views of 16121
unique training frames, not 72000 independent images. No validation frames are
added to training. The first partial block can overlap the following full pass.

The controller stops after three validation checks without >0.001 probe AP gain,
the frame budget, a STOP marker, or the training wall allowance. Its persistent
deadline begins after acquiring the GPU lease. Two of the eight hours are
reserved for confirmation/export, with soft overruns for active phases. Every
measured block gets an immutable rung checkpoint; receipts recover completed
blocks without training them again. The original public-best weights remain
untouched.

The strongest measured rung must improve the current champion on the identical
probe with positive paired family evidence before full confirmation. Promotion
and CSV export require the unseen-family native AND full official tiled gates;
only then is the shared local champion updated. The desired Kaggle score is
0.28+, not an observed result or a guarantee. No Kaggle submission is automatic.
