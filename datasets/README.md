# Dataset Layout

```text
datasets/
├── raw/
│   ├── generator_sources/
│   └── legacy_archive/
├── prepared/
│   ├── architectures/
│   ├── cifake/
│   ├── corpus/
│   ├── legacy_archive/
│   ├── highres_corpus/
│   ├── modern_corpus/
│   ├── modern_v2/
│   └── multires/
├── evaluation/
│   └── internet/
└── fixtures/
    └── corpus_smoke/
```

- `raw/` contains source material retained in its original or legacy form.
- `prepared/` contains corpora ready for training and validation.
- `prepared/legacy_archive/` is the flattened, trainable form of the raw archive.
- `evaluation/` contains independent datasets reserved for model evaluation.
- `fixtures/` contains small corpora used by tests and smoke checks.

The separate `dataset/` directory contains dataset builder and downloader scripts. `features/` contains derived model inputs rather than source data. Paths recorded in manifests are relative to their corpus roots.
