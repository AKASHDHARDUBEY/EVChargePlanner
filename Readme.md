# EVision

### Intelligent EV Charging Demand Prediction & Agentic Infrastructure Planning System

EVision is an AI-driven analytics and decision-support system designed to **predict electric vehicle (EV) charging demand** using historical charging data and to **generate intelligent infrastructure planning recommendations** using an agentic AI assistant.

This project is developed as part of an academic initiative and follows strict constraints such as **no paid APIs**, **public deployment**, and **open-source / free-tier tools only**.

---

## Project Objectives

* Predict EV charging demand using historical charging session data
* Identify peak demand periods and high-load charging stations
* Visualize trends and forecasts through an interactive web interface
* Extend predictions into an **agentic AI assistant** that provides:

  * Infrastructure expansion recommendations
  * Load balancing and scheduling insights
  * Data-driven planning reports with references

---

## 🧩 Project Structure

The project is divided into **two major milestones**:

### Milestone 1: EV Charging Demand Prediction (Mid-Sem)

* Data preprocessing and feature engineering
* ML / time-series forecasting models
* Evaluation using MAE and RMSE
* Interactive UI for predictions and trend visualization

### Milestone 2: Agentic AI Infrastructure Planning Assistant (End-Sem)

* Analysis of predicted demand patterns
* Retrieval of EV infrastructure planning guidelines
* Agentic reasoning workflow using LLMs
* Structured infrastructure planning reports

---

## Dataset

**Source:** Dryad Digital Repository
🔗 [https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z](https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z)

The dataset contains historical EV charging session data including:

* Charging timestamps and duration
* Energy consumption (kWh)
* Charging station identifiers
* Temporal usage patterns

---

## System Architecture

```
CSV Upload
   ↓
Data Preprocessing & Feature Engineering
   ↓
Demand Forecasting Model (ML / Time-Series)
   ↓
Demand Insights & Peak Analysis
   ↓
Agentic AI Planning Assistant
   ↓
Infrastructure Recommendation Report
```

---

## Tech Stack

### Data & ML

* Python
* pandas, NumPy
* scikit-learn
* statsmodels (optional)

### Agentic AI (Milestone 2)

* Open-source / free-tier LLMs
* LangGraph (agent workflow)
* Retrieval-Augmented Generation (optional)

### UI & Visualization

* Streamlit (primary)
* matplotlib / seaborn / plotly

### Hosting

* Hugging Face Spaces / Streamlit Community Cloud (free tier)

---

## Model Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

These metrics are used to evaluate demand prediction accuracy.

---

## Features

* Upload EV charging datasets (CSV)
* Predict future charging demand
* Visualize demand trends and peak usage periods
* Identify high-load charging stations
* Generate structured infrastructure planning recommendations
* Handle incomplete or noisy data gracefully

---

## Example Agent Output (Milestone 2)

* Charging demand summary
* Identification of high-load locations
* Infrastructure expansion suggestions
* Scheduling and load-balancing strategies
* Supporting references and assumptions

---

## How to Run

1. Install dependencies:
```bash

pip install -r requirements.txt
```

2. Place the `volume.csv` file in the `data/` folder

3. Run the application:
```bash
streamlit run app.py
```

4. Open http://localhost:8501 in your browser

---

## Team

This project is developed by a team of **4 students**, with clear ownership across:

* Data preprocessing
* ML & forecasting
* UI & visualization
* Agentic AI & planning

---

## Deployment

**Important:** Localhost-only demos are not accepted.

The final application will be:

* Publicly accessible
* Hosted on a free-tier cloud platform

Deployment link will be added here once live.

---

## Demo & Submission

* 🎥 Demo video (5–7 minutes)
* Public application URL
* Complete GitHub repository

---
