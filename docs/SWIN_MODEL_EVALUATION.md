# SwinV2 checkpoint evaluation

This is the audit record for every Swin checkpoint currently present under
`models/`. It separates two questions that should not be conflated:

1. **Can the saved file be served?** The current audit answers this for every
   file by strictly loading the state dictionary and running a 512×512 forward
   pass.
2. **How well does an independent weight state detect generated images?** The
   historical benchmark artifacts answer this for the saved epoch runs. Those
   figures are useful evidence, but are not one new, directly comparable
   leaderboard because the older runs used different transforms and union-test
   sample sizes.

## Current audit (2 September 2026)

Run:

```bash
.venv/bin/python diag_swin_checkpoint_audit.py
MPLCONFIGDIR=/tmp/truthlens-mpl .venv/bin/python -m pytest \
  tests/test_swin_manifest.py tests/test_verify_swin_dataset.py \
  tests/test_swin_train_eval.py tests/test_predict.py -q
```

Result: **22/22 checkpoint files passed** strict restoration and a finite
`[1, 2]` classifier output; **72/72 regression tests passed**. The machine-
readable audit is [`results/metrics/swin_checkpoint_audit.json`](../results/metrics/swin_checkpoint_audit.json).

There are only **18 unique weight states**. The following files are bytewise
identical at the state-dictionary level and must not be counted as independent
models:

| Canonical checkpoint | Identical file(s) | Meaning |
|---|---|---|
| `swin_v2_512_ep012.pth` | `swin_resume_swin_v2_512.pth` | Final baseline epoch and its resume state |
| `swin_v2_512_improved_ep005.pth` | `swin_v2_tiny_512_improved.pth` | Improved run’s best-export alias |
| `swin_v2_512_newdata_ep003.pth` | `swin_v2_tiny_512_newdata.pth` | New-data best-export alias |
| `swin_v2_tiny_512_hardstyles.pth` | `swin_v2_tiny_512.pth` | Hard-styles export and default-named alias |

Two further files are active training-resume states, not released model
snapshots: `swin_resume_swin_v2_512_improved.pth` (epoch 7) and
`swin_resume_swin_v2_512_improved_backup_20260827_195528.pth` (epoch 1). They
passed compatibility testing, but have no published holdout score and should
not be selected for the app.

## Quality evidence by training run

All figures below use the fixed `P(FAKE) >= 0.5` decision rule. “Bal.” means
balanced accuracy. `H` is the quarantined 611-image holdout, `U` is a balanced
union-test sample, and `W` is the 351-image Internet Wild set. The holdout is
the most useful guard against source and generator leakage.

### Improved run

These were measured together with JPEG quality-88 recompression and 500
real + 500 fake union-test images. Source:
[`results/metrics/improved_swin_benchmark.json`](../results/metrics/improved_swin_benchmark.json).

| Checkpoint | H AUC | H Bal. | U Bal. | W Bal. | H recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|
| `improved_ep001` | 0.9139 | 85.48% | 90.20% | 66.81% | 76.23% |
| `improved_ep002` | 0.9331 | 85.76% | 90.20% | 82.26% | 77.21% |
| `improved_ep003` | 0.9229 | 84.88% | 87.60% | **85.93%** | 76.62% |
| `improved_ep004` | 0.9327 | 85.28% | 85.00% | 72.21% | 82.71% |
| `improved_ep005` / `swin_v2_tiny_512_improved` | 0.9261 | 85.28% | **92.80%** | 77.13% | 80.75% |
| `improved_ep006` | **0.9415** | **89.40%** | 92.30% | 83.40% | **85.07%** |

`improved_ep006` is the preferred improved-run checkpoint: it has the best
quarantined-holdout AUC, balanced accuracy, and constrained-FPR recall. Epoch
3 is the better choice only if Internet Wild balanced accuracy is the sole
selection criterion.

### Baseline and hard-styles run

These six checkpoints were measured together in the earlier baseline protocol
with a 250-real + 250-fake union-test sample. Source:
[`results/metrics/swin_model_comparison.json`](../results/metrics/swin_model_comparison.json).

| Checkpoint | H AUC | H Bal. | U Bal. | W Bal. | Composite* |
|---|---:|---:|---:|---:|---:|
| `hardstyles` / `swin_v2_tiny_512` | **0.9418** | 84.50% | **90.60%** | 83.33% | **89.03** |
| `ep008` | 0.9278 | **88.51%** | 85.80% | 83.57% | 87.02 |
| `ep009` | 0.9267 | **88.51%** | 88.80% | **87.69%** | 88.99 |
| `ep010` | 0.9015 | 78.81% | 88.80% | 73.26% | 84.21 |
| `ep011` | 0.9370 | 87.83% | 87.00% | 80.27% | 86.72 |
| `ep012` / baseline resume | 0.9255 | 84.69% | 85.20% | 80.03% | 85.46 |

\*The historical composite combines holdout AUC (35%), Wild balanced accuracy
(25%), union balanced accuracy (25%), and holdout recall at 5% FPR (15%). It
is valid within this table only—not against the improved-run table.

### New-data random-initialization run

The four completed epochs were tested on the full 2,500-image union test and
the same holdout/Wild sets. Source:
[`results/metrics/swin_epochs_test_summary.json`](../results/metrics/swin_epochs_test_summary.json).

| Checkpoint | H AUC | H Bal. | U Bal. | W Bal. | Assessment |
|---|---:|---:|---:|---:|---|
| `newdata_ep001` | 0.6603 | 50.00% | 50.00% | 50.00% | Collapsed to FAKE at threshold 0.5 |
| `newdata_ep002` | 0.4264 | 50.00% | 50.00% | 50.00% | Do not serve |
| `newdata_ep003` / `swin_v2_tiny_512_newdata` | 0.4432 | 50.00% | 50.00% | 50.00% | Do not serve |
| `newdata_ep004` | 0.4499 | 50.00% | 50.00% | 50.00% | Do not serve |

The new-data run predicts FAKE for every tested image at the default threshold;
its superficially high raw holdout accuracy is caused by that split being
83.3% fake. Balanced accuracy exposes the failure.

## Selection and reporting rules

- Serve `models/swin_epochs/swin_v2_512_improved_ep006.pth` if selecting from
  the improved training run; it is the strongest held-out result in that run.
- Do not use the new-data checkpoints, resume snapshots, or the duplicate file
  names as independent alternatives.
- Do not call any checkpoint an overall “champion” across the two historical
  benchmark files. Their preprocessing and union-test sample sizes differ.
- Before changing the app default, rerun all distinct deployable states through
  one versioned benchmark command with identical transforms and fixed row IDs;
  write the new result to `results/metrics/` and update this document.
