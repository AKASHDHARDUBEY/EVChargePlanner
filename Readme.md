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

## 📂 Repository Structure

```text
EVChargePlanner/
│
├── data/
│   └── volume.csv                 # Historical EV charging datasets
│
├── knowledge/
│   └── ev_guidelines.txt          # RAG Knowledge Base (PDF/Text context)
│
├── src/
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   ├── model.py                   # Random Forest & Linear Regression models
│   ├── visualization.py           # Charts and interactive plots
│   ├── rag.py                     # ChromaDB Vector Store for document retrieval 
│   ├── agent.py                   # LangGraph AI Workflow (Analyzer, Planner, Reviewer)
│   ├── llm.py                     # Groq LLM integration
│   └── report.py                  # Generates the final PDF exports
│
├── app.py                         # Main Streamlit interface
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🚀 How to Run

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your `volume.csv` file in the `data/` folder.

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Open http://localhost:8501 in your browser.

5. **For Milestone 2 (AI Agent):** Enter your free **Groq API Key** in the sidebar configuration to unlock the LangGraph Agentic Planner and Interactive QA Chatbot.

---

## 👥 Team Roles

This project is developed by a team of **4 students**, with clear ownership across:
* **Amanjeet (Agent Architect):** LangGraph workflow logic and state management.
* **Shubhi Kumari (Knowledge Engineer):** RAG vector database and Prompt Engineering.
* **Ayush Kumar Pandey (Integration Lead):** ML outputs to LLM pipeline and error handling.
* **Akash Dhar Dubey (UI & DevOps):** Streamlit frontend, PDF extensions, and cloud deployment.

---

## ☁️ Deployment

**Important:** Localhost-only demos are not accepted.
The final application is successfully deployed and publicly accessible via Streamlit Community Cloud:
🔗 **[Live Application: EV Charge Planner](https://ev-planner.streamlit.app/)**

---

## 🎥 Demo & Submission

* 🎥 Demo video (5–7 minutes) demonstrating both the Models and the AI Agent.
* **Public application URL:** [https://ev-planner.streamlit.app/](https://ev-planner.streamlit.app/)
* Complete GitHub repository
