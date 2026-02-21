#!/usr/bin/env python3
"""
Stress Test — All API Endpoints
GET  /
GET  /api/v1/health
POST /api/v1/predict
POST /api/v2/predict
POST /api/risk-timeline
GET  /api/roads
GET  /api/login
GET  /metrics
"""

import asyncio
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, quantiles
from typing import Optional

import httpx

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────
BASE_URL       = "https://localhost"
AUTH           = ("admin", "password")
TOTAL_REQUESTS = 20000
CONCURRENCY    = 40

# Pool de villes
CITIES_POOL = [
    "Yvrac", "Ambarès-et-Lagrave", "Lormont", "Bordeaux", "Mérignac",
    "Pessac", "Talence", "Bègles", "Cenon", "Floirac",
    "Villenave-d'Ornon", "Gradignan", "Le Bouscat", "Eysines", "Bruges",
    "Blanquefort", "Parempuyre", "Saint-Médard-en-Jalles", "Bassens", "Carbon-Blanc",
]

# Exemple features v1 (à adapter à votre modèle)
SAMPLE_FEATURES = {
    "heure": 14, "jour": 3, "mois": 2, "an": 2026,
    "lum": 1, "agg": 1, "int": 1, "atm": 1, "col": 1,
    "catr": 2, "circ": 2, "nbv": 2, "prof": 1, "plan": 1,
    "surf": 1, "infra": 0, "situ": 1, "vma": 50,
    "sexe": 1, "trajet": 1, "secu1": 1, "catv": 7,
    "obsm": 0, "choc": 1, "manv": 1,
}


# ── Payload generators ──────────────────────────────────────────
def payload_v2() -> dict:
    cities    = random.sample(CITIES_POOL, random.randint(1, 5))
    random_dt = datetime(2026, 1, 1) + timedelta(
        days=random.randint(0, 89),
        hours=random.randint(0, 23)
    )
    return {"cities": cities, "timestamp": random_dt.strftime("%Y-%m-%dT%H:%M")}


def payload_v1() -> dict:
    f = SAMPLE_FEATURES.copy()
    f["heure"] = random.randint(0, 23)
    f["jour"]  = random.randint(1, 7)
    f["mois"]  = random.randint(1, 12)
    f["vma"]   = random.choice([30, 50, 70, 90, 110, 130])
    return {"features": f}


# ── Endpoint definitions ────────────────────────────────────────
ENDPOINTS = [
    {"name": "GET /",                  "method": "GET",  "path": "/",                  "weight": 1},
    {"name": "GET /api/v1/health",     "method": "GET",  "path": "/api/v1/health",     "weight": 3},
    {"name": "POST /api/v1/predict",   "method": "POST", "path": "/api/v1/predict",    "weight": 10, "payload": payload_v1},
    {"name": "POST /api/v2/predict",   "method": "POST", "path": "/api/v2/predict",    "weight": 30, "payload": payload_v2},
    {"name": "POST /api/risk-timeline","method": "POST", "path": "/api/risk-timeline", "weight": 5},
    {"name": "GET /api/roads",         "method": "GET",  "path": "/api/roads",         "weight": 5},
    {"name": "GET /api/login",         "method": "GET",  "path": "/api/login",         "weight": 2},
    {"name": "GET /metrics",           "method": "GET",  "path": "/metrics",           "weight": 2},
]


# ── Result ──────────────────────────────────────────────────────
@dataclass
class Result:
    req_id:     int
    endpoint:   str
    method:     str
    status:     int
    elapsed_ms: float
    payload:    Optional[dict] = None
    error:      str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


# ── Single request ──────────────────────────────────────────────
async def do_request(client: httpx.AsyncClient, req_id: int, ep: dict, sem: asyncio.Semaphore) -> Result:
    payload = ep["payload"]() if "payload" in ep else None
    url     = BASE_URL + ep["path"]
    async with sem:
        start = time.perf_counter()
        try:
            if ep["method"] == "GET":
                resp = await client.get(url)
            else:
                resp = await client.post(url, json=payload)
            elapsed = (time.perf_counter() - start) * 1000
            return Result(req_id, ep["name"], ep["method"], resp.status_code, elapsed, payload)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Result(req_id, ep["name"], ep["method"], 0, elapsed, payload, error=str(e))


# ── Formatting ──────────────────────────────────────────────────
def color_status(status: int) -> str:
    if 200 <= status < 300: return f"\033[92m{status}\033[0m"
    if status == 0:         return f"\033[91mERR\033[0m"
    if status < 500:        return f"\033[93m{status}\033[0m"
    return f"\033[91m{status}\033[0m"


def print_result(r: Result):
    ep_short = r.endpoint.split(" ", 1)[-1]
    extra = ""
    if r.payload:
        if "cities" in r.payload:
            extra = f"  | {r.payload['timestamp']}  {', '.join(r.payload['cities'])}"
        elif "features" in r.payload:
            extra = f"  | heure={r.payload['features'].get('heure')} vma={r.payload['features'].get('vma')}"
    line = f"  [{r.req_id:>3}] {color_status(r.status)} — {r.elapsed_ms:>7.1f}ms  {ep_short:<28}{extra}"
    if r.error:
        line += f"\n         \033[91m↳ {r.error[:120]}\033[0m"
    print(line)


# ── Summary ─────────────────────────────────────────────────────
def print_summary(results: list, global_ms: float):
    total_ok = sum(1 for r in results if r.ok)
    rps      = len(results) / (global_ms / 1000)

    print("\n\033[96m" + "=" * 76 + "\033[0m")
    print("\033[96m   RÉSUMÉ GLOBAL\033[0m")
    print("\033[96m" + "=" * 76 + "\033[0m")
    print(f"  Total requêtes : {len(results)}  |  "
          f"\033[92mSuccès : {total_ok}\033[0m  |  "
          f"\033[91mÉchecs : {len(results)-total_ok}\033[0m  |  "
          f"Durée : {global_ms:.0f}ms  |  Débit : {rps:.2f} req/sec")

    # Par endpoint
    by_ep = defaultdict(list)
    for r in results:
        by_ep[r.endpoint].append(r)

    header = f"\n  {'Endpoint':<32} {'N':>5} {'OK':>5} {'Err':>5} {'Avg':>8} {'P95':>8} {'P99':>8} {'Max':>8}"
    print("\033[96m" + header + "\033[0m")
    print("\033[96m" + "  " + "-" * 74 + "\033[0m")

    for ep_name in sorted(by_ep.keys()):
        grp   = by_ep[ep_name]
        times = [r.elapsed_ms for r in grp]
        ok    = sum(1 for r in grp if r.ok)
        err   = len(grp) - ok
        p95   = quantiles(times, n=100)[94] if len(times) >= 2 else times[0]
        p99   = quantiles(times, n=100)[98] if len(times) >= 2 else times[0]
        ec    = "\033[91m" if err > 0 else "\033[92m"
        reset = "\033[0m"
        print(
            f"  {ep_name:<32} {len(grp):>5} "
            f"{ec}{ok:>5}{reset} {ec}{err:>5}{reset} "
            f"{mean(times):>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {max(times):>7.1f}ms"
        )

    print("\033[96m" + "=" * 76 + "\033[0m")

    if len(results) == total_ok:
        print("  \033[92m✔ Tous les appels ont réussi\033[0m")
    else:
        errors = [r for r in results if not r.ok]
        print(f"  \033[91m✘ {len(errors)} erreur(s) — détail (max 10) :\033[0m")
        for r in errors[:10]:
            print(f"    [{r.req_id}] {r.endpoint}  status={r.status or 'ERR'}  {r.error}")

    print("\033[96m" + "=" * 76 + "\033[0m\n")


# ── Main ─────────────────────────────────────────────────────────
async def main():
    total_w   = sum(ep["weight"] for ep in ENDPOINTS)
    task_pool = [ep for ep in ENDPOINTS for _ in range(ep["weight"])]
    tasks_def = [random.choice(task_pool) for _ in range(TOTAL_REQUESTS)]

    print("\n\033[96m" + "=" * 76 + "\033[0m")
    print("\033[96m   Stress Test — Tous les endpoints\033[0m")
    print("\033[96m" + "=" * 76 + "\033[0m")
    print(f"  Base URL    : {BASE_URL}")
    print(f"  Total req   : {TOTAL_REQUESTS}  (répartis par poids)")
    print(f"  Concurrency : {CONCURRENCY}\n")
    print(f"  {'Endpoint':<32} {'Poids':>6}  {'~Req':>6}")
    print(f"  {'-'*46}")
    for ep in ENDPOINTS:
        print(f"  {ep['name']:<32} {ep['weight']:>6}  {round(TOTAL_REQUESTS * ep['weight'] / total_w):>6}")
    print("\033[96m" + "-" * 76 + "\033[0m\n")

    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(
        auth=AUTH,
        verify=False,
        timeout=15.0,
        limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY),
    ) as client:
        coros   = [do_request(client, i + 1, ep, sem) for i, ep in enumerate(tasks_def)]
        results = []
        start   = time.perf_counter()
        for coro in asyncio.as_completed(coros):
            r = await coro
            print_result(r)
            results.append(r)
        global_ms = (time.perf_counter() - start) * 1000

    results.sort(key=lambda r: r.req_id)
    print_summary(results, global_ms)


if __name__ == "__main__":
    asyncio.run(main())
