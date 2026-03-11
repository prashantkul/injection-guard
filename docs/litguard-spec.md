# litguard

LitServe-based model serving platform for prompt injection detection models, with a React monitoring UI. Designed as a DGX Spark playbook.

## Stack
- **Backend**: [LitServe](https://litserve.ai) — ML model serving framework by Lightning AI
- **Frontend**: React + Vite + Tailwind CSS — monitoring dashboard
- **Models**: HuggingFace classification models served as OpenAI-compatible endpoints

## Backend Requirements

### LitServe Server
- Serve `deepset/deberta-v3-base-injection` as the default model
- Support loading multiple HuggingFace text-classification models via config
- Expose OpenAI-compatible `/v1/chat/completions` endpoint so existing OpenAI SDK clients work without changes
- Map classification labels (INJECTION/LEGIT) to structured JSON responses:
  `{"label": "injection"|"benign", "score": 0.0-1.0, "confidence": 0.0-1.0}`
- Enable LitServe's built-in request batching for GPU throughput
- Add `/health`, `/models`, and `/metrics` endpoints
- `/metrics` returns: requests/sec, avg latency, model load status, GPU utilization, classification distribution (injection vs benign counts)
- Support `DEVICE=cuda:0` or `cpu` via env var
- Config via `config.yaml`:
  ```yaml
  models:
    - name: deberta-injection
      hf_model: deepset/deberta-v3-base-injection
      device: cuda:0
      batch_size: 32
    - name: protectai-injection
      hf_model: protectai/deberta-v3-base-prompt-injection-v2
      device: cuda:0
      batch_size: 32
  port: 8234
  ```

### API Contract
- `POST /v1/chat/completions` — OpenAI-compatible (so injection-guard library works)
- `GET /health` — `{"status": "ok", "models_loaded": ["deberta-injection"]}`
- `GET /models` — list loaded models
- `GET /metrics` — server stats (Prometheus-compatible preferred)
- `GET /api/history` — recent classification results (last 1000, stored in-memory)

## Frontend Requirements

### React Monitoring Dashboard
- Real-time metrics: requests/sec, avg latency, GPU util (poll /metrics every 2s)
- Classification pie chart: injection vs benign distribution
- Recent requests table: timestamp, input preview (truncated), label, score, latency
- Model status cards: loaded models, their device, batch size
- Dark mode, responsive layout
- Connect to backend via env var `VITE_API_URL=http://localhost:8234`

## Project Structure
```
litguard/
├── README.md
├── pyproject.toml          # uv/pip, deps: litserve, transformers, torch
├── config.yaml
├── src/
│   └── server/
│       ├── __init__.py
│       ├── app.py          # LitServe app + API definition
│       ├── models.py       # Model loading + inference logic
│       └── metrics.py      # In-memory metrics collector
├── ui/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── MetricsPanel.tsx
│       │   ├── ClassificationChart.tsx
│       │   ├── RequestsTable.tsx
│       │   └── ModelStatus.tsx
│       └── hooks/
│           └── useMetrics.ts
├── Dockerfile              # Multi-stage: Python backend + React static
├── docker-compose.yaml     # Backend + UI services
└── playbook/
    └── README.md           # DGX Spark deployment instructions
```

## DGX Spark Playbook
- Docker Compose based deployment
- Auto-download models on first run
- GPU passthrough via nvidia-container-toolkit
- Expose port 8234 (API) and 3000 (UI)
- Include setup.sh that handles nvidia-docker prereqs
- Reference implementation: [NVIDIA/dgx-spark-playbooks#65](https://github.com/NVIDIA/dgx-spark-playbooks/pull/65)

## Key Constraints
- Use `uv` for Python dependency management
- TypeScript strict mode for React
- No authentication needed (runs on private network)
- Models cached in Docker volume to avoid re-download
- Keep it simple — this is a single-node deployment, not k8s
