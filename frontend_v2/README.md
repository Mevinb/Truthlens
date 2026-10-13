# TruthLens V2 frontend

This folder contains a complete Streamlit redesign that leaves the original
`app/` frontend unchanged. It imports the existing inference layer from
`app/inference.py`, so model loading, preprocessing, prediction, and evidence
generation still use the same code as the current application.

## Run

From the project root:

```bash
./frontend_v2/run.sh
```

The launcher starts Streamlit from this folder so the local theme configuration
in `frontend_v2/.streamlit/config.toml` is applied.

## Included behavior

- Single and batch image upload
- Held-out sample images
- All existing model choices
- Confidence review threshold
- Optional ResNet18 attention maps
- Provenance and watermark review
- Frequency and compression evidence
- JSON and CSV exports
- Live benchmark data from the existing result files
- Light and dark appearance modes

The model is still being trained. The interface therefore presents every output
as a screening result rather than proof of authenticity.
