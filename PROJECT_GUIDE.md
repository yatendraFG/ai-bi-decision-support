# Project Guide — ye project kaise bana, aur aap agla project khud kaise banaoge

## 1. Sach-sach: maine kaise banaya?

- **Koi ready-made / pretrained model nahi liya.** Har model is project ke data par scratch se train hua hai:
  scikit-learn (gradient boosting, logistic regression, KMeans, Isolation Forest) aur TensorFlow/Keras
  (LSTM, autoencoder).
- **Development ke time notebook use nahi kiya.** Seedha `src/` mein `.py` files likhi, terminal se chalayi,
  output padha. Jab koi sawal tha (jaise "log target better hai ya Poisson?") toh ek chhota throwaway
  script chala kar dono compare kiye, jo jeeta wahi `src/` mein gaya.
- **Notebooks baad mein banaye** (`notebooks/01_…`, `02_…`) taaki aap har step chala kar dekh sako ki
  kya hua, kyun hua, aur kaunse experiments fail hue.

### Notebook vs `.py` — kab kya?

| Notebook (`.ipynb`) | Script (`src/*.py`) |
|---|---|
| Data explore karna, plots, ek idea jaldi try karna | Final, dobara chalne wala code |
| Seekhna / samjhana / presentation | Dashboard, API, agent isi ko import karte hain |
| Cells upar-neeche chalne se state gadbad ho sakti hai | Har baar upar se neeche same result |
| Git mein review karna mushkil | Saaf diff, test karna aasaan |

**Professional workflow:** notebook mein explore → jo kaam kare use function bana kar `src/` mein daalo →
notebook ab `src/` ko import kare. Is project ke notebooks bhi yahi karte hain (`from src.ml import forecasting`).

---

## 2. Folder structure — har cheez kyun hai

```
ai-bi-system/
├── data/
│   ├── raw/            ← INPUT. Company ka data (yahan generated). Kabhi edit nahi karte.
│   └── processed/      ← OUTPUT. Models ke results (forecast, churn scores, segments, anomalies)
├── models/             ← Trained models (.joblib = sklearn, .keras = TensorFlow). Dashboard/agent yahin se load karenge
├── reports/            ← Metrics JSON + summary.md. "Model kitna accha hai" ka saboot
├── notebooks/          ← Seekhne / explore karne ke liye
├── src/                ← Asli code (Python package)
│   ├── config.py       ← Saare paths + constants EK jagah. Path badalna ho toh sirf yahan
│   ├── data/           ← Step 2: data laana / banana
│   ├── ml/             ← Steps 3-4: classic ML
│   ├── dl/             ← Step 5: deep learning
│   ├── nlp/            ← Step 6: ticket sentiment/category + insights
│   ├── rag/            ← Step 7: document Q&A (RAG)
│   ├── agent/          ← Step 8: tools + planner + report (AI agent)
│   ├── api/            ← Step 9: REST API (FastAPI)
├── app/                ← Step 9: Streamlit dashboard
├── run_pipeline.py     ← Ek command se sab kuch, sahi order mein
├── requirements.txt    ← Kaunsi libraries chahiye
├── README.md           ← Project kya hai + kaise chalayein
└── PROJECT_GUIDE.md    ← Ye file
```

**Golden rule:** `raw` data → `src` code → `models` + `processed` + `reports`. Code kabhi raw data nahi badalta,
isliye kabhi bhi sab delete karke `python run_pipeline.py` se sab dobara ban jaata hai.

### Data flow

```
data/raw/sales.csv ─┐
data/raw/customers.csv ─┼─► src/ml/features.py ──► churn.py ──────► churn_scores.csv ─┐
data/raw/tickets.csv ─┘          │                                                       ├─► customer_360.csv
                                 └──────────────► segmentation.py ─► customer_segments.csv ┘
data/raw/sales.csv ──► forecasting.py ──► sales_forecast.csv
                 └───► dl/lstm_forecast.py (same folds) ──► lstm_forecast_metrics.json
                 └───► dl/anomaly.py ──► order_anomalies.csv
data/raw/tickets.csv ──► nlp/ticket_classifier.py ──► ticket_predictions.csv + models
                                    └──► nlp/ticket_insights.py (+ churn_scores) ──► urgent_tickets.csv
reports/*.json ──► data/make_company_docs.py ──► data/docs/*.pdf ──► rag/rag_system.py ──► answer + citation
all of the above ──► agent/tools.py ◄── agent/planner.py (or Claude) ◄── question
                          └──► agent/agent.py ──► reports/agent/*.md (findings, what it means, actions)
everything ──► app/dashboard.py (people, browser)   and   src/api/main.py (other software, JSON)
```

`customer_360.csv`, `order_anomalies.csv`, `urgent_tickets.csv` aur RAG — sab Step 8 ke agent ke "tools" ban gaye.

---

## 3. Har file kya karti hai

| File | Input | Kaam | Output |
|---|---|---|---|
| `src/config.py` | — | Paths, random seed, horizon | — |
| `src/data/generate_data.py` | — | Realistic synthetic data (trend, festive peak, churn signals) | `data/raw/*.csv` |
| `src/ml/features.py` | raw CSVs + snapshot date | Customer features **sirf snapshot se pehle ke data se** | DataFrame |
| `src/ml/forecasting.py` | sales | Daily revenue/region, 30-day forecast, 12-fold backtest vs baselines | forecast CSV, model, metrics |
| `src/ml/churn.py` | features | 90-day churn, LR vs GB, test on unseen snapshot, risk reasons | churn_scores, model, metrics |
| `src/ml/segmentation.py` | features | RFM + KMeans, naam + action, churn ke saath join | segments, customer_360 |
| `src/dl/lstm_forecast.py` | sales | Keras LSTM, Step 3 ke same folds par compare + ensemble (MSE loss: 6.9% vs GB 7.9%) | lstm model, metrics |
| `src/dl/anomaly.py` | sales + customers | Keras autoencoder vs Isolation Forest, injected anomalies se test | order_anomalies, model, metrics |
| `src/data/ticket_text.py` | category + sentiment | Realistic ticket text (typos, Hinglish, sarcasm, vague, bina tone) | text column |
| `src/nlp/ticket_classifier.py` | tickets | Keyword vs TF-IDF vs TF-IDF+priority vs Keras BiGRU; confidence se review queue | ticket models, predictions, metrics |
| `src/nlp/ticket_insights.py` | tickets + churn scores | Category trend, complaint phrases, aaj call karne wali list | urgent_tickets.csv, insights JSON |
| `src/data/make_company_docs.py` | reports JSON | Sample company PDFs (policies, SOP, Q3 review with model numbers) | `data/docs/*.pdf` |
| `src/rag/rag_system.py` | PDFs | Chunking, 5 retrievers, section/sentence answers, optional Claude, abstain | `RAG().ask(q)` |
| `src/rag/evaluate_rag.py` | 37 + 6 test questions | Methods compare, threshold + answer mode choose | rag_config.json, rag_metrics.json |
| `src/rag/ask.py` | — | Interactive Q&A (`ask_documents.bat`) | — |
| `src/agent/tools.py` | Steps 3-7 outputs | 9 tools, har ek `{summary, data, source}` + JSON-schema specs | — |
| `src/agent/planner.py` | question | Hinglish normalise, entities, intent scores, recipes (diagnose/actions/overview) | plan |
| `src/agent/agent.py` | plan | Tools chalao → findings + "what this means" + actions; optional Claude tool-use loop | reports/agent/*.md |
| `src/agent/evaluate_agent.py` | 30 unseen questions | Tool precision/recall, entities, errors | agent_metrics.json |
| `app/dashboard.py` | processed CSVs + tools + models | 6 pages: Overview, Sales, Customers, Support, Anomalies, Ask AI | browser app |
| `src/api/main.py` | same | 12 REST endpoints (Swagger docs at /docs) | JSON |
| `src/api/test_api.py` | — | 15 endpoint checks incl. bad input (404/422) | pass/fail |
| `run_pipeline.py` | — | Sab steps order mein + business summary | `reports/summary.md` |

### Har model file ka same pattern (ise copy karo)

```python
"""Docstring: business sawal, approach, evaluation, outputs."""
CONSTANTS                     # snapshot dates, horizon...
def load / build_features()   # data -> features
def build_model()             # model definition
def evaluate()                # metrics jo business samjhe
def main():
    data -> baseline -> model -> same test par compare -> best choose
    -> final model sab data par -> save model + predictions + metrics JSON
if __name__ == "__main__": main()
```

Isliye har file alag bhi chalti hai: `python -m src.ml.churn`

---

## 4. Agla project khud banane ka step-by-step template

1. **Business sawal likho** — "kaun customer chhodega, agle 90 din mein?" (target + time window clear)
2. **Folder banao** — upar wala structure copy karo, `config.py` pehle likho
3. **Data load + EDA** — notebook mein. Trend, missing values, distributions, target ka balance
4. **Split decide karo** — time-series/customers: **time-based split**, kabhi random nahi.
   Test set ko end tak mat chhuo
5. **Baseline banao** — naive rule ya logistic regression. Iska number likh lo
6. **Features** — har feature ke liye pucho: "kya ye prediction ke time pata hoga?" Nahi → leakage
7. **Model try karo** — same test par baseline se compare. Ek se zyada model, validation par choose
8. **Debug** — predictions ka average vs actual (bias), probabilities sahi hain? (calibration),
   kis type ke cases fail ho rahe hain?
9. **`src/` mein shift karo** — function banao, `main()` likho, model + metrics save karo
10. **Pipeline + README** — ek command se sab chale; README mein results honest likho

---

## 5. Is project mein pakdi gayi galtiyan (aapke kaam aayengi)

| Galti | Kaise pakdi | Fix |
|---|---|---|
| Generated data mein 2026 mein sales gir rahi thi (naye customers band) | Quarterly revenue print kiya | Signups END tak continue |
| Forecast har mahine kam aa raha tha | Per-fold error ka sign dekha | log target hataya, Poisson + level ratio |
| Sirf Q3 par test → model naive se haar raha tha | Ek hi season par test tha | 12-month rolling backtest |
| Churn probabilities 10x badi | "Revenue at risk" ₹10 Cr (total ka 15%!) — number absurd laga | `class_weight="balanced"` hataya |
| Best customers ka naam "low value" | Segment table padha | Naming rules rank-based banaye |
| pandas 2.3 par crash | Aapke laptop par chalaya | Version-safe code |
| LSTM training slow (13 × 43 sec) | Ek fold time kiya | Halka model + fold cache (resume) |
| LSTM har mahine kam forecast (9.5% error) | Per-fold error ka sign — phir se minus | MAE loss → MSE loss (6.9%, sabse accha model) |
| NLP models 100% accurate | `t.text.nunique()` → sirf 59 sentences! | Realistic text generator |
| Sentiment 78% par atak gaya | Error analysis: tone wale tickets 100%, bina tone 48% | Bada model nahi, nayi jaankari: priority feature → 85% |
| Keras = TF-IDF (0.003 fark) | Validation table | Fark < 0.005 → simple model chuno |
| RAG jawab sirf 76% sahi, jabki sahi section 97% mil raha tha | Retrieval aur answer alag-alag naapa | Bottleneck = sentence picker → poora section dikhao (89%) |
| Ek asli sawal "not found" ho gaya | Threshold round(…, 2) upar chala gaya | Threshold hamesha neeche round karo |
| Paraphrase sawal fail ("money back" vs "refund") | Failures list | Test-set ke synonyms NAHI jode (cheating); upgrade = embeddings/LLM |
| Agent report sirf numbers, matlab nahi | Report padhi: West tickets company se KAM negative the | "What this means": company average se comparison, "ye wajah nahi hai" |
| Policy sawal par ticket tool bhi chala | Unseen test (87%) | Logic fix → 90% (honestly: test dekh kar); vocab misses jaan-boojh kar chhode |
| Dashboard 203 high-risk, reports 213 | Screenshot dekh kar numbers compare kiye | Ek hi source (churn_scores) poore dashboard mein |
| KPI "Rs 689.3 l…" kat raha tha, chart zero se shuru | Asli browser screenshot | Unit label mein, sirf poore mahine |

**Sabse bada sabak:** har number ko padh kar pucho "kya ye sense banata hai?" Accuracy se zyada galtiyan
sanity checks se pakdi gayi.

---

## Step 9: Dashboard aur API - kab kya?

| | Dashboard (`app/dashboard.py`) | API (`src/api/main.py`) |
|---|---|---|
| Kiske liye | Insaan (manager, sales, support) | Doosra software (CRM, website, Power BI, mobile app) |
| Output | Charts, tables, chat - browser mein | JSON |
| Chalana | `run_dashboard.bat` → http://localhost:8501 | `run_api.bat` → http://127.0.0.1:8000/docs |

**Streamlit ke 3 concepts:**
- Har click par poori script upar se neeche dobara chalti hai. Isliye data load ko `@st.cache_data` (CSV/tables)
  aur models ko `@st.cache_resource` (RAG, joblib) se cache karte hain, warna har click slow hoga.
- Chat history jaisi cheezein `st.session_state` mein rakhte hain (re-run ke baad bhi bachi rehti hain).
- Ek hi number ka ek hi source: dashboard ne pehle churn count galat file se gina (203 vs 213) — screenshot dekh kar pakda.

**FastAPI ke 3 concepts:**
- Function ke upar `@app.get("/forecast")` = ek URL. Type hints (`region: Literal["North", ...]`) se input
  apne aap check hota hai: galat region → 422 error, model tak pahunchta hi nahi.
- `/docs` page apne aap banta hai — wahin se har endpoint try kar sakte hain.
- `TestClient` se bina server chalaye API test hoti hai (`python -m src.api.test_api`).

## Step 10: GitHub + Docker - saath mein kaise kaam karte hain

```
laptop: code + run_pipeline.py (training)  --git push-->  GitHub (code, history, portfolio)
                                                              |
                                              GitHub Actions: tests -> docker build -> run check
                                                              |
                                              GitHub Packages: ready image (ghcr.io/...)
                                                              |
                                  koi bhi server / laptop: docker run ...  (same result everywhere)
```

| Cheez | Kya karti hai |
|---|---|
| `requirements-deploy.txt` | Sirf dashboard + API ki libraries. TensorFlow/Jupyter nahi -> image ~1 GB chhoti. scikit-learn version pinned (pickled model usi version se load hota hai) |
| `Dockerfile` | Recipe: python:3.11-slim -> libraries -> code + models -> non-root user -> dashboard start. Libraries pehle copy hoti hain taaki code badalne par rebuild fast ho (layer cache) |
| `docker-compose.yml` | Ek hi image se 2 containers: dashboard (8080) + API (8000) |
| `.dockerignore` | Notebooks, logs, LSTM cache image mein na jaayein |
| `.github/workflows/ci.yml` | Har push par tests + image build; main par image publish |

**Zaroori sabak:** "mere laptop par chalta hai" kaafi nahi. Slim requirements ko ek saaf environment mein test kiya
(bina TensorFlow) - tabhi pata chala ki dashboard/API ko TensorFlow ki zarurat nahi.

## 6. Environment (aapke laptop par set ho chuka hai)

Har project ka **apna alag environment** hona chahiye, taaki ek project ki libraries doosre ko na todein.
Is project ke liye conda environment `ai-bi` bana hai (Python 3.11, pandas, scikit-learn, TensorFlow, Jupyter).

| File | Kaam |
|---|---|
| `setup_env.bat` | Double-click → `ai-bi` env banata hai, `requirements.txt` install karta hai, Jupyter kernel register karta hai. Log: `setup_log.txt` |
| `run_project.bat` | Double-click → poora pipeline `ai-bi` env mein chalata hai. Log: `run_log.txt`, result: `reports\\summary.md` |

Manually (Anaconda Prompt mein) yahi kaam:

```
conda create -n ai-bi python=3.11 -y
conda activate ai-bi
cd %USERPROFILE%\\Desktop\\ai-bi-system
pip install -r requirements.txt ipykernel
python -m ipykernel install --user --name ai-bi --display-name "Python (ai-bi)"
python run_pipeline.py
```

- **VS Code:** `Ctrl+Shift+P` → "Python: Select Interpreter" → `ai-bi` choose karo.
- **Jupyter:** notebook kholo → Kernel → Change kernel → **Python (ai-bi)**.
- **Naya library chahiye?** `conda activate ai-bi` → `pip install <name>` → `requirements.txt` mein bhi likho.
- **Agla project:** naya env banao (`conda create -n mera-project python=3.11`), is wale ko mat chhuo.

## 7. Commands

```bash
pip install -r requirements.txt
python run_pipeline.py                 # Steps 2-8 sab + API check
streamlit run app/dashboard.py         # dashboard
uvicorn src.api.main:app --reload      # API
python run_pipeline.py --skip-generate # apna data use karo
python run_pipeline.py --skip-dl       # sirf ML (fast)
python -m src.dl.lstm_forecast         # sirf ek step
jupyter notebook notebooks/            # walkthrough notebooks
```
