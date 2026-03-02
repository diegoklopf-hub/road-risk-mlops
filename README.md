# MLOps Pipeline for Road Accident Data Analysis

**Program:** Machine Learning Engineer – Expert in Artificial Intelligence Engineering  
**Institution:** Liora / Mines ParisTech Executive Education  

**Authors:**
- Julie PINTO
- Diego KLOPFENSTEIN
- Yasser BELAIDI
- Yves BRU

**Cohort Lead:** Maria DOMENZAIN ACEVEDO  
**Project Mentor:** Antoine Fradin  

---

## Overview

This project provides a complete MLOps pipeline to predict the severity of road accidents within a targeted geographic area.  

It powers a graphical interface used by a fire station located in Bassens (33, Gironde, France) to identify the most at-risk roads in the coming hours and visualize a 24-hour risk timeline.

The system relies on BAAC data (French Annual Road Traffic Injury Accident Database) from 2019 to 2024, which is cleaned and encoded before training an XGBoost model to generate contextualized predictions.

Predictions are computed using a combination of:
- Structural variables (speed limit, road width, infrastructure, etc.) defined and updated by the user through the interface
- Real-time weather data (https://openweathermap.org/)
- Temporal context (date/time)

---

## Objectives

- Provide the Bassens fire station with predictions of severe accident risk in the coming hours, localized by road.
- Display the Top 5 roads with the highest severe accident probability in the interface.
- Generate a 24-hour risk timeline to assist with operational planning.
- Combine structural variables (speed limit, width, infrastructure, etc.) defined and updated by the user with real-time weather data and temporal context (date/time) to refine predictions.

---

## Prerequisites

- **Python 3.9+**
- **Virtual environment** (venv/conda)
- **Nginx SSL Certificates**
  - `deployments/nginx/certs/nginx.crt`
  - `deployments/nginx/certs/nginx.key`
- **User creation**

---

## Project Initialization

On first use, run the full initialization:

```bash
make init
