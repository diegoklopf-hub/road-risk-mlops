# 🚒 Road Risk MLOps Platform  
### Production-Ready ML System for Road Accident Risk Prediction

**Program:** Machine Learning Engineer – AI Engineering Expert  
**Institution:** Liora / Mines ParisTech Executive Education  

**Authors:**  
Julie Pinto • Diego Klopfenstein • Yasser Belaidi • Yves Bru  

---

# 🎯 Executive Summary

This project delivers a **production-grade MLOps platform** designed to predict severe road accident risk in a targeted geographical area (Bassens, France).

The system supports operational decision-making for a fire station by:

- Identifying the **Top 5 highest-risk roads**
- Providing a **24-hour risk timeline**
- Combining **static road infrastructure data**, **real-time weather conditions**, and **temporal context**

The platform is built following modern **MLOps best practices**, including:

- Modular pipeline architecture
- Data validation & schema enforcement
- Model versioning
- Dockerized services
- API deployment
- Monitoring-ready infrastructure

---

# 🏗 System Architecture

The platform is fully containerized and orchestrated using Docker.

Core components:

- **Airflow** – Pipeline orchestration  
- **MLflow** – Experiment tracking & model management  
- **FastAPI** – Real-time inference API  
- **XGBoost** – Risk prediction model  
- **Nginx** – Reverse proxy & SSL  
- **Prometheus + Grafana** – Monitoring stack  

High-level architecture:

BAAC Data (2019–2024)
↓
Data Validation (Schema.yaml)
↓
Data Cleaning & Feature Engineering
↓
Resampling & Training (XGBoost)
↓
Model Artifacts (MLflow)
↓
FastAPI Inference Service
↓
Web Interface (S.A.V.E.R.)


---

# 🚀 Key Technical Highlights

## ✅ End-to-End MLOps Pipeline

- Multi-step modular pipeline
- Config-driven architecture (`config.yaml`)
- YAML-based feature selection
- Fail-fast validation strategy
- Reproducible training

---

## ✅ Strict Data Governance

- Centralized schema definition
- Automatic validation at every critical stage
- Missing / extra column detection
- Type enforcement
- Execution status tracking

This ensures **production reliability and auditability**, key in regulated environments (e.g., Switzerland).

---

## ✅ Model Training & Explainability

- XGBoost classifier
- Hyperparameter tuning via configurable `param_grid`
- SHAP explainability integration
- Feature persistence & artifact management

Artifacts loaded at API startup:

- `best_model.joblib`
- `features.joblib`
- `one_hot_encoder.joblib`
- `shap_explainer.joblib`

Fail-fast startup if artifacts are missing.

---

## ✅ Production-Ready API

- FastAPI-based inference engine
- Health check endpoint
- Structured JSON responses
- Real-time weather integration (OpenWeather API)
- UTC-safe timestamp handling
- Feature engineering at inference time

Example inference request:

```json
{
  "cities": ["Bassens"],
  "timestamp": "2026-02-10T22:00:00Z"
}

## ✅ Monitoring & Observability

Centralized logging

Prometheus metrics collection

Grafana dashboards

Structured logs for debugging

API health endpoint

Designed with scalability and production monitoring in mind.

##🧠 Business Impact

This platform enables:

Data-driven emergency resource allocation

Preventive risk visualization

Real-time operational awareness

Evidence-based planning

It demonstrates the ability to bridge:

Data Engineering → Machine Learning → DevOps → Production Deployment

##  🛠 Tech Stack

Languages

Python 3.9+

ML & Data

XGBoost

Pandas

Scikit-learn

SHAP

MLOps & Orchestration

Airflow

MLflow

Docker / Docker Compose

YAML-driven configuration

API & Backend

FastAPI

Nginx

Basic Auth

Monitoring

Prometheus

Grafana

##  🔍 Testing Strategy

Unit tests (API & components)

Integration tests

Debug mode execution

Logging validation

Run tests:

make unit-test
make int-test

##  📂 Project Structure (Simplified)
src/
 ├── data_processing/
 ├── modeling/
 ├── pipeline/
 ├── api/
deployments/
 ├── mlflow/
 ├── prometheus/
 ├── grafana/
 └── nginx/
docker-compose.yml

##  ⚙️  How to Run
make init
make start-project

Stop:

make stop-project

##  📌 What This Project Demonstrates

✔ Ability to design production ML systems
✔ Clean, modular, scalable architecture
✔ Strong MLOps practices
✔ Data validation & governance awareness
✔ Monitoring & observability mindset
✔ Explainability integration
✔ Docker-based deployment
