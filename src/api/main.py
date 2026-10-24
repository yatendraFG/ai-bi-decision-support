"""
Step 9a: REST API - the same models and tools, available to any app (website, CRM, mobile, Excel/Power BI).

Run:   uvicorn src.api.main:app --reload          (or double-click run_api.bat)
Docs:  http://127.0.0.1:8000/docs                 (interactive Swagger page - try every endpoint there)

Endpoints
  GET  /health                         service + which model files are present
  GET  /forecast?region=West           next-30-day forecast
  GET  /sales/trend?region=&category=  last quarter vs previous + drivers
  GET  /churn?region=&segment=&top_n=  high-risk customers
  GET  /segments                       customer segments
  GET  /tickets/insights?category=     complaint trends + themes
  GET  /tickets/urgent?region=&top_n=  who to call today
  GET  /anomalies?region=&top_n=       suspicious orders
  GET  /customers/{customer_id}        one customer's profile
  POST /tickets/classify               {"text": "...", "priority": "High"} -> category, sentiment, needs_review
  POST /ask/documents                  {"question": "..."} -> answer + sources (RAG, Step 7)
  POST /ask/agent                      {"question": "..."} -> markdown report (agent, Step 8)
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from functools import lru_cache
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import MODELS
from src.agent import tools as T

app = FastAPI(title="Kaveri Retail AI BI API", version="1.0",
              description="Forecasts, churn, segments, support tickets, anomalies, document Q&A and the AI agent.")

Region = Optional[Literal["North", "South", "East", "West"]]


def clean(obj):
    """Make pandas/numpy values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def tool(name, **kw):
    out = T.TOOLS[name](**{k: v for k, v in kw.items() if v is not None})
    if out.get("data") is None:
        raise HTTPException(404, out["summary"])
    return clean(out)


@app.get("/health")
def health():
    files = ["sales_forecaster.joblib", "churn_model.joblib", "segmentation.joblib", "ticket_sentiment.joblib",
             "ticket_category.joblib", "rag_config.json"]
    present = {f: (MODELS / f).exists() for f in files}
    return {"status": "ok" if all(present.values()) else "degraded - run run_project.bat", "models": present}


@app.get("/forecast")
def forecast(region: Region = None):
    return tool("sales_forecast", region=region)


@app.get("/sales/trend")
def sales_trend(region: Region = None,
                category: Optional[Literal["Electronics", "Furniture", "Office Supplies", "Clothing"]] = None):
    return tool("sales_trend", region=region, category=category)


@app.get("/churn")
def churn(region: Region = None, segment: Optional[str] = None, top_n: int = Query(10, ge=1, le=200)):
    return tool("churn_risk", region=region, segment=segment, top_n=top_n)


@app.get("/segments")
def segments():
    return tool("customer_segments")


@app.get("/tickets/insights")
def ticket_insights(category: Optional[Literal["Delivery", "Billing", "Product Quality", "Technical", "Refund"]] = None,
                    region: Region = None):
    return tool("ticket_insights", category=category, region=region)


@app.get("/tickets/urgent")
def urgent(region: Region = None, top_n: int = Query(10, ge=1, le=200)):
    return tool("urgent_calls", region=region, top_n=top_n)


@app.get("/anomalies")
def anomalies(region: Region = None, top_n: int = Query(10, ge=1, le=200)):
    return tool("order_anomalies", region=region, top_n=top_n)


@app.get("/customers/{customer_id}")
def customer(customer_id: str):
    return tool("customer_lookup", customer_id=customer_id.upper())


# ---------------------------------------------------------------- model endpoints
class Ticket(BaseModel):
    text: str = Field(..., min_length=3, examples=["Refund abhi tak nahi aaya, 3 hafte ho gaye!!"])
    priority: Literal["Low", "Medium", "High"] = "Medium"


@lru_cache
def ticket_models():
    return {t: joblib.load(MODELS / f"ticket_{t}.joblib") for t in ("sentiment", "category")}


@app.post("/tickets/classify")
def classify(ticket: Ticket):
    X = pd.DataFrame([{"text": ticket.text, "priority": ticket.priority}])
    out, review = {}, False
    for task, b in ticket_models().items():
        p = b["model"].predict_proba(X)[0]
        i = int(p.argmax())
        out[task] = {"label": str(b["model"].classes_[i]), "confidence": round(float(p[i]), 3),
                     "all": {str(c): round(float(v), 3) for c, v in zip(b["model"].classes_, p)}}
        review |= p[i] < b["review_threshold"]
    out["needs_review"] = bool(review)
    return out


class Question(BaseModel):
    question: str = Field(..., min_length=3, examples=["refund card mein kitne din mein aayega?"])


@lru_cache
def rag():
    from src.rag.rag_system import RAG
    return RAG()


@app.post("/ask/documents")
def ask_documents(q: Question):
    return rag().ask(q.question)


@app.post("/ask/agent")
def ask_agent(q: Question):
    from src.agent.agent import ask
    r = ask(q.question)
    return {"report": r["report"], "tools_used": [c["tool"] for c in r["plan"]["calls"]] if "plan" in r else r.get("tools_used"),
            "saved_to": str(r.get("path")) if r.get("path") else None}
