# AI Business Intelligence & Decision Support System

Company ke sales, customers, support tickets aur documents analyze karke actionable
recommendations dene wala AI platform.

Naye ho? Pehle `PROJECT_GUIDE.md` padhiye (project kaise bana + agla project khud kaise banayein),
phir `notebooks/` ke walkthrough chalaiye.

## Roadmap (numbered steps)

| Step | Module | Status |
|---|---|---|
| 1 | Project setup (structure, config) | Done |
| 2 | Business dataset generator (sales, customers, tickets) | Done |
| 3 | Sales forecasting (gradient boosting, backtested vs baselines) | Done |
| 4 | Churn prediction + customer segmentation (RFM + KMeans) | Done |
| 5 | DL (Keras): LSTM forecasting + autoencoder anomaly detection | Done |
| 6 | NLP: ticket sentiment + category, review queue, urgent call list | Done |
| 7 | RAG: company PDFs / policies Q&A with citations + "not found" | Done |
| 8 | AI agent: question -> right tools -> report with findings, meaning, actions | Done |
| 9 | Streamlit dashboard (6 pages) + FastAPI (12 endpoints) | Done |
| 10 | Docker + GitHub Actions (tests, image build, GitHub Packages) | Done |

## Folder structure

```
ai-bi-system/
├── data/raw/            # input CSVs (sales, customers, tickets)
├── data/docs/           # company PDFs for RAG (samples: Kaveri Retail policies, SOP, Q3 review)
├── data/processed/      # model outputs (forecast, churn, segments, customer_360, anomalies, tickets, urgent list)
├── models/              # saved models (.joblib = sklearn, .keras = TensorFlow)
├── reports/             # metrics JSON + summary.md + agent/ (saved agent reports)
├── notebooks/
│   ├── 01_ml_walkthrough_steps_2_4.ipynb   # step-by-step: EDA -> baseline -> model -> debug
│   ├── 02_dl_walkthrough_step5.ipynb       # LSTM + autoencoder, piece by piece
│   ├── 03_nlp_walkthrough_step6.ipynb      # tickets: baseline -> TF-IDF -> error analysis -> actions
│   ├── 04_rag_walkthrough_step7.ipynb      # PDFs -> chunks -> retrieval -> answers -> abstain
│   └── 05_agent_walkthrough_step8.ipynb    # tools -> planner -> report -> evaluation -> Claude option
├── src/
│   ├── config.py        # paths, constants
│   ├── data/generate_data.py, ticket_text.py, make_company_docs.py
│   ├── ml/features.py, forecasting.py, churn.py, segmentation.py
│   ├── dl/lstm_forecast.py, anomaly.py
│   ├── nlp/ticket_classifier.py, ticket_insights.py
│   ├── rag/rag_system.py, evaluate_rag.py, ask.py
│   ├── agent/tools.py, planner.py, agent.py, evaluate_agent.py, ask_agent.py
│   └── api/main.py, test_api.py   # REST API
├── app/dashboard.py     # Streamlit dashboard
├── run_pipeline.py      # runs Steps 2-8 end to end + API check
├── setup_env.bat / run_project.bat / run_dashboard.bat / run_api.bat
├── open_notebooks.bat / ask_documents.bat / ask_agent.bat              # Windows one-click
├── requirements.txt         # full (training): TensorFlow, Jupyter, ...
├── requirements-deploy.txt  # slim (dashboard + API only) - used by Docker
├── Dockerfile, docker-compose.yml, .dockerignore
├── docker_up.bat / docker_down.bat
├── .github/workflows/ci.yml # GitHub Actions: tests + Docker image
├── tests/smoke_dashboard.py
├── README.md
└── PROJECT_GUIDE.md
```

## Run it

**Windows (one-click):** `setup_env.bat` (ek baar / nayi library aaye tab) → `run_project.bat` (project chalao) → `open_notebooks.bat` (Jupyter) → `ask_documents.bat` (documents se sawal poocho) → `ask_agent.bat` (business sawal, poori report) → **`run_dashboard.bat`** (browser mein dashboard) → `run_api.bat` (REST API, docs: http://127.0.0.1:8000/docs).

**Manually (Anaconda Prompt):**

```bash
conda create -n ai-bi python=3.11 -y
conda activate ai-bi
pip install -r requirements.txt
python run_pipeline.py             # Steps 2-8 (~2-5 min; first LSTM backtest without cache ~10 min)
python run_pipeline.py --skip-dl   # no TensorFlow: skips Step 5 + the Keras NLP model
```

Har step alag se bhi chala sakte hain:

```bash
python -m src.data.generate_data
python -m src.ml.forecasting
python -m src.ml.churn
python -m src.ml.segmentation
python -m src.dl.lstm_forecast     # finished folds are cached in reports/lstm_folds_*/ (resumable)
python -m src.dl.anomaly
python -m src.nlp.ticket_classifier
python -m src.nlp.ticket_insights
python -m src.data.make_company_docs
python -m src.rag.evaluate_rag
python -m src.rag.ask               # interactive document Q&A
python -m src.agent.evaluate_agent
python -m src.agent.ask_agent       # interactive AI agent
python -m src.agent.agent "West ki sales kyun gir rahi hai?"   # one question -> report
streamlit run app/dashboard.py      # dashboard  -> http://localhost:8501
uvicorn src.api.main:app --reload   # API        -> http://127.0.0.1:8000/docs
python -m src.api.test_api          # API checks (no server needed)
```

Data dobara generate karne ke baad `reports/lstm_folds_*` folder delete kar dijiye, taaki LSTM naye data par train ho.

## Docker (dashboard + API ek command mein)

Docker Desktop install hona chahiye. Phir `docker_up.bat` double-click, ya:

```bash
docker compose up --build -d     # dashboard http://localhost:8080 , API http://localhost:8000/docs
docker compose down              # stop
```

Image mein sirf deploy ki cheezein hain (`requirements-deploy.txt`, bina TensorFlow/Jupyter) + trained models
aur outputs. Training laptop par hoti hai (`run_pipeline.py`); naye models ke baad `docker compose up --build -d`
dobara chalaiye.

## GitHub + CI

Har push par GitHub Actions (`.github/workflows/ci.yml`):
1. API ke 15 checks + dashboard ke 6 pages test karta hai
2. Docker image banata hai, chala kar check karta hai
3. `main` branch par image GitHub Packages mein publish: `ghcr.io/<user>/ai-bi-decision-support:latest`

Kisi bhi machine par (Docker ho toh): `docker run -p 8080:8501 ghcr.io/<user>/ai-bi-decision-support:latest`

## Apna real data use karna

`data/raw/` mein same columns wali CSVs rakh dijiye (schema `src/data/generate_data.py` ke docstring mein)
aur `python run_pipeline.py --skip-generate` chalaiye.
**Real data ko public GitHub repo mein mat daaliye** - `.gitignore` mein bataye lines uncomment karke repo private rakhiye.

Apne PDFs `data/docs/` mein daaliye (sample PDFs hata sakte hain). Apne sawal + sahi jawab
`src/rag/evaluate_rag.py` ki `QUESTIONS` list mein jodiye, taaki accuracy apne documents par naap sakein.
