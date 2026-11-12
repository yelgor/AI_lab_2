# Training throughput and measured physical batch

This changes the training loop itself. It is separate from candidate quality
selection and does not establish a new Kaggle score. Keep learned c1 weights,
optimizer moments, full-data coverage and the robust selection guards.

## Runtime changes

The previous loop checked loss finiteness on the CPU and called `loss.item()`
for every minibatch. Both can synchronize CUDA. The shared `train_batches`
loop now accumulates detached FP64 scalar statistics on device, checks their
finiteness at each optimizer-update boundary BEFORE stepping weights, and
reads loss for progress reports/end of phase. A final short accumulation group
is normalized just as before. Invalid losses never step that update. AMP's
gradient-overflow handling and gradient clipping remain in place.

CPU equivalence tests compare parameters, AdamW moments/counters and total loss
bit for bit against the retained `per_batch` reference, including partial groups.
The CUDA equivalence test is available but skipped in the CPU-only web executor.
The new code does not promise bit-identical GPU behavior without that check.
No LR, training sample, objective, precision or architecture change is implicit
in the loss-readback optimization. `runtime.loss_sync=per_batch` remains an
explicit reference option for controlled recipes.

For the five-minibatch/accumulation-two test, loss scalar reads fall from five
to one and finite checks from five to three. These are ONLY loss bookkeeping
polls. RT-DETR matching, AMP and other model operations can still synchronize;
this is not a measured percentage speedup. Unused extra loader batches are no
longer fetched after a bounded block has reached its limit.

History and pending-validation receipts now include `train_timing`: actual
training pipeline seconds, frames/s, host loader wait, attempted optimizer
update boundaries and loss-readback counts. The pipeline excludes validation
and serialization. Host wait can overlap GPU work and cannot be interpreted as
an additive GPU-idle fraction. AMP can skip an update on overflow. A separately
recorded pre-validation checkpoint duration is available when measured.

## Measure on the actual laptop

Run after the healthy owner releases the shared GPU lease. The default command
exits if busy; `--wait` queues without stopping that owner. The script does not
save modified model weights or export a submission.

```bash
.venv/bin/python -u scripts/benchmark_training.py \
  --work "$CVLAB_LOCAL_WORK" \
  --source "$CVLAB_LOCAL_WORK/source_c1d7fcc83dde/best.pt" \
  --config "$CVLAB_LOCAL_WORK/runs/rtdetr_c1_continuation_train_selection/config.json" \
  --batch-sizes 1 2 --updates 20 --warmup-updates 3 --repeats 2 --vram-gib 6
```

Use the actual configured source/config paths if they differ. With no config,
the benchmark uses the established full-data recipe, not a new random model.
It requires available RAM, avoids another compute process, caps its own CUDA
allocator, and does not retry a batch size that already exceeded that budget.
Every case begins from its own copy of the same learned weights, optimizer and
RNG. A dense-GT stress batch checks denoising/activation memory, then source
weights/moments/scaler/RNG are restored before warmup and measurement. It tests
the real training loop with old/new loss-readback modes in alternating order.
Power/temperature snapshots and peak allocation stay in the local JSON.

Results: `WORK/benchmarks/training_*.json` (gitignored). OOM cases cannot become
recommendations. At least two repeats and >3% median improvement over the
batch-one reference are required for a compute recommendation; this is a noise
guard, not a confidence claim. Cold-cache/thermal bias can remain. A fastest
case with no accepted gain is still reported honestly. The CPU unit benchmark
does not establish GPU speed.

## Apply physical batch only after measurement

The full-data launcher accepts `--batch-size 2` in a **new plan name** after
measurement. It transforms the established full-data `batch=1, accumulation=2`
recipe to `batch=2, accumulation=1`, preserving two frames per update and the
LR/budget. The requested batch must divide the established effective batch.
An existing plan cannot silently switch batch size on resume.

Validation remains batch one for comparable evaluation and cache provenance;
the receipt identity now includes the actual inference batch. Full-epoch
coverage checks count actual frames, not minibatches, including the last odd
frame. The source checkpoint is never overwritten.

Equal effective batch does not prove equal RT-DETR training: object-count loss
normalization, denoising and stochastic behavior can depend on physical batch.
Run the same held-out class/family gates before accepting it. Do not claim an AP
gain from a throughput test or start concurrent models on the 8 GiB laptop.
Mid-epoch cursor recovery and a hard full-epoch wall limit remain separate work.
