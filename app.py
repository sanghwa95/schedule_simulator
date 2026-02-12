# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from simulator import run_simulation

app = FastAPI(title="Schedule Simulator", version="1.0.0")


# ---------------------------
# Request schema (validation)
# ---------------------------
class SimRequest(BaseModel):
    # ❌ le=200_000 제거
    iters: int = Field(10000, ge=1, description="Number of iterations")
    seed: int = Field(1, description="RNG seed")
    top: int = Field(5, ge=1, le=50, description="How many top/bottom schedules to return")

    # wake-up range (hours)
    min_wake_hour: int = Field(0, ge=0, le=12, description="Minimum wake-up hour (0–12)")
    max_wake_hour: int = Field(9, ge=0, le=12, description="Maximum wake-up hour (0–12)")

    # overtime rules
    ot_rate: int = Field(30000, ge=0, le=1_000_000, description="Overtime pay (won/hour)")
    ot_max: int = Field(5, ge=0, le=12, description="Max overtime hours/day")

    # time resolution
    slot: int = Field(30, ge=5, le=60, description="Slot size in minutes (grid resolution)")


# ---------------------------
# Health check
# ---------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------
# Simulation endpoint
# ---------------------------
@app.post("/api/simulate")
def simulate(req: SimRequest):
    """
    Run simulation and return JSON result.
    Automatically clamp iterations to 200,000.
    """

    # ✅ 자동 보정
    MAX_ITERS = 200_000
    safe_iters = min(req.iters, MAX_ITERS)

    result = run_simulation(
        iters=safe_iters,
        seed=req.seed,
        top=req.top,
        wake_min=req.min_wake_hour * 60,
        wake_max=req.max_wake_hour * 60,
        ot_rate=req.ot_rate,
        ot_max=req.ot_max,
        slot=req.slot,
    )

    return JSONResponse(result)


# ---------------------------
# Convenience: basic root
# ---------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/index.html")
