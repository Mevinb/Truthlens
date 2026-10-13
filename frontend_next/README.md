# TruthLens Next frontend

This is a separate Next.js and Tailwind CSS frontend for the existing TruthLens model stack. It does not replace or modify `app/` or `frontend_v2/`.

## Run locally

From the repository root:

```bash
cd frontend_next
../.venv/bin/python -m pip install -r api/requirements.txt
npm install
chmod +x run.sh
./run.sh
```

Open [http://localhost:3000](http://localhost:3000). The script starts the Python model API on port 8000 and the Next.js app on port 3000.

## Run the services separately

```bash
cd frontend_next
../.venv/bin/python -m uvicorn api.server:app --reload --port 8000
```

In another terminal:

```bash
cd frontend_next
npm run dev
```

Set `TRUTHLENS_API_URL` if the Python service runs somewhere other than `http://127.0.0.1:8000`.

## What is connected

- Single and batch image upload with no fixed queue limit
- Existing model registry from `app.inference`
- Existing shared prediction path from `src.predict`
- Optional Grad-CAM for supported ResNet models
- Live repository benchmark and resolution metrics
- Result JSON export, forensic signals, and provenance details
- Light and dark themes
