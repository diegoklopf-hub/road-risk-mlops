"""
Traffic Generator – S.A.V.E.R. Stress Test
Utilise requests + ThreadPoolExecutor (pas de dépendance externe à Airflow).
"""
from __future__ import annotations

import random
import time
import urllib3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, quantiles
from typing import Optional

import requests

# Désactive les warnings SSL (verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config par défaut ───────────────────────────────────────────
DEFAULT_BASE_URL = "https://nginx"
DEFAULT_AUTH = ("admin", "password")
DEFAULT_TOTAL_REQUESTS = 20_000
DEFAULT_CONCURRENCY = 40

# ── Pool de villes ──────────────────────────────────────────────
CITIES_POOL = [
    "Yvrac", "Ambarès-et-Lagrave", "Lormont", "Bordeaux", "Mérignac",
    "Pessac", "Talence", "Bègles", "Cenon", "Floirac",
    "Villenave-d'Ornon", "Gradignan", "Le Bouscat", "Eysines", "Bruges",
    "Blanquefort", "Parempuyre", "Saint-Médard-en-Jalles", "Bassens", "Carbon-Blanc",
]

# ── Features v1 (template) ─────────────────────────────────────
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
    cities = random.sample(CITIES_POOL, random.randint(1, 5))
    random_dt = datetime(2026, 1, 1) + timedelta(
        days=random.randint(0, 89),
        hours=random.randint(0, 23),
    )
    return {"cities": cities, "timestamp": random_dt.strftime("%Y-%m-%dT%H:%M")}


def payload_v1() -> dict:
    features = SAMPLE_FEATURES.copy()
    features["heure"] = random.randint(0, 23)
    features["jour"] = random.randint(1, 7)
    features["mois"] = random.randint(1, 12)
    features["vma"] = random.choice([30, 50, 70, 90, 110, 130])
    return {"features": features}


# ── Endpoint definitions ────────────────────────────────────────
def build_endpoints() -> list[dict]:
    return [
        {"name": "GET /",                   "method": "GET",  "path": "/",                  "weight": 1},
        {"name": "GET /api/v1/health",      "method": "GET",  "path": "/api/v1/health",     "weight": 3},
        {"name": "POST /api/v1/predict",    "method": "POST", "path": "/api/v1/predict",    "weight": 10, "payload": payload_v1},
        {"name": "POST /api/v2/predict",    "method": "POST", "path": "/api/v2/predict",    "weight": 30, "payload": payload_v2},
        {"name": "POST /api/risk-timeline", "method": "POST", "path": "/api/risk-timeline", "weight": 5},
        {"name": "GET /api/roads",          "method": "GET",  "path": "/api/roads",         "weight": 5},
        {"name": "GET /api/login",          "method": "GET",  "path": "/api/login",         "weight": 2},
        {"name": "GET /metrics",            "method": "GET",  "path": "/metrics",           "weight": 2},
    ]


# ── Result dataclass ────────────────────────────────────────────
@dataclass
class Result:
    req_id: int
    endpoint: str
    method: str
    status: int
    elapsed_ms: float
    payload: Optional[dict] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


# ── Single request (synchrone) ──────────────────────────────────
def do_request(
    session: requests.Session,
    req_id: int,
    endpoint: dict,
    base_url: str,
) -> Result:
    payload = endpoint["payload"]() if "payload" in endpoint else None
    url = base_url + endpoint["path"]
    start = time.perf_counter()
    try:
        if endpoint["method"] == "GET":
            resp = session.get(url, timeout=15)
        else:
            resp = session.post(url, json=payload, timeout=15)
        elapsed = (time.perf_counter() - start) * 1000
        return Result(req_id, endpoint["name"], endpoint["method"], resp.status_code, elapsed, payload)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return Result(req_id, endpoint["name"], endpoint["method"], 0, elapsed, payload, error=str(exc))


# ── Formatting ──────────────────────────────────────────────────
def color_status(status: int) -> str:
    if 200 <= status < 300:
        return f"\033[92m{status}\033[0m"
    if status == 0:
        return "\033[91mERR\033[0m"
    if status < 500:
        return f"\033[93m{status}\033[0m"
    return f"\033[91m{status}\033[0m"


def format_result(result: Result) -> str:
    ep_short = result.endpoint.split(" ", 1)[-1]
    extra = ""
    if result.payload:
        if "cities" in result.payload:
            extra = f"  | {result.payload['timestamp']}  {', '.join(result.payload['cities'])}"
        elif "features" in result.payload:
            extra = (
                f"  | heure={result.payload['features'].get('heure')} "
                f"vma={result.payload['features'].get('vma')}"
            )
    line = f"  [{result.req_id:>5}] {color_status(result.status)} — {result.elapsed_ms:>7.1f}ms  {ep_short:<28}{extra}"
    if result.error:
        line += f"\n          \033[91m↳ {result.error[:120]}\033[0m"
    return line


def print_summary(results: list[Result], global_ms: float):
    total_ok = sum(1 for r in results if r.ok)
    rps = len(results) / (global_ms / 1000) if global_ms > 0 else 0

    print("\n\033[96m" + "=" * 76 + "\033[0m")
    print("\033[96m   RÉSUMÉ GLOBAL\033[0m")
    print("\033[96m" + "=" * 76 + "\033[0m")
    print(
        f"  Total requêtes : {len(results)}  |  "
        f"\033[92mSuccès : {total_ok}\033[0m  |  "
        f"\033[91mÉchecs : {len(results) - total_ok}\033[0m  |  "
        f"Durée : {global_ms:.0f}ms  |  Débit : {rps:.2f} req/sec"
    )

    by_ep = defaultdict(list)
    for r in results:
        by_ep[r.endpoint].append(r)

    header = f"\n  {'Endpoint':<32} {'N':>5} {'OK':>5} {'Err':>5} {'Avg':>8} {'P95':>8} {'P99':>8} {'Max':>8}"
    print("\033[96m" + header + "\033[0m")
    print("\033[96m  " + "-" * 74 + "\033[0m")

    for ep_name in sorted(by_ep.keys()):
        grp = by_ep[ep_name]
        times = [r.elapsed_ms for r in grp]
        ok = sum(1 for r in grp if r.ok)
        err = len(grp) - ok
        p95 = quantiles(times, n=100)[94] if len(times) >= 2 else times[0]
        p99 = quantiles(times, n=100)[98] if len(times) >= 2 else times[0]
        ec = "\033[91m" if err > 0 else "\033[92m"
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


# ── Main entry point ────────────────────────────────────────────
def generate_traffic(
    total_requests: int = DEFAULT_TOTAL_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    base_url: str = DEFAULT_BASE_URL,
    auth: tuple[str, str] = DEFAULT_AUTH,
):
    endpoints = build_endpoints()
    total_weight = sum(ep["weight"] for ep in endpoints)
    task_pool = [ep for ep in endpoints for _ in range(ep["weight"])]
    tasks_def = [random.choice(task_pool) for _ in range(total_requests)]

    print("\n\033[96m" + "=" * 76 + "\033[0m")
    print("\033[96m   Stress Test — Tous les endpoints (requests + ThreadPool)\033[0m")
    print("\033[96m" + "=" * 76 + "\033[0m")
    print(f"  Base URL    : {base_url}")
    print(f"  Total req   : {total_requests}  (répartis par poids)")
    print(f"  Concurrency : {concurrency}\n")
    print(f"  {'Endpoint':<32} {'Poids':>6}  {'~Req':>6}")
    print(f"  {'-' * 46}")
    for ep in endpoints:
        expected = round(total_requests * ep["weight"] / total_weight)
        print(f"  {ep['name']:<32} {ep['weight']:>6}  {expected:>6}")
    print("\033[96m" + "-" * 76 + "\033[0m\n")

    session = requests.Session()
    session.auth = auth
    session.verify = False
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=concurrency,
        pool_maxsize=concurrency,
        max_retries=0,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    results: list[Result] = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(do_request, session, i + 1, ep, base_url): i
            for i, ep in enumerate(tasks_def)
        }
        for future in as_completed(futures):
            result = future.result()
            print(format_result(result))
            results.append(result)

    global_ms = (time.perf_counter() - start) * 1000

    session.close()
    results.sort(key=lambda r: r.req_id)
    print_summary(results, global_ms)