# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from simulator import run_simulation

app = FastAPI(title="Time = Money Simulator", version="1.0.0")


# ---------------------------
# Request schema (validation)
# ---------------------------
class SimRequest(BaseModel):
    iters: int = Field(10000, ge=1, le=200_000, description="Number of iterations")
    seed: int = Field(1, description="RNG seed")
    top: int = Field(5, ge=1, le=50, description="How many top/bottom schedules to return")

    # wake-up range (minutes from 00:00)
    wake_min: int = Field(360, ge=0, le=24 * 60, description="Minimum wake-up time in minutes")
    wake_max: int = Field(540, ge=0, le=24 * 60, description="Maximum wake-up time in minutes")

    # overtime rules
    ot_rate: int = Field(30000, ge=0, le=1_000_000, description="Overtime pay (won/hour)")
    ot_max: int = Field(4, ge=0, le=12, description="Max overtime hours/day")

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
    """
    result = run_simulation(**req.model_dump())
    return JSONResponse(result)


# ---------------------------
# Convenience: basic root
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Time = Money Simulator API",
        "endpoints": ["/api/health", "/api/simulate", "/docs"],
    }
