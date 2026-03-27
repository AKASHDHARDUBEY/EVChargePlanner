import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import (
    load_volume_data, prepare_features, get_feature_columns,
    split_data, get_peak_hours, get_peak_days, get_summary_stats
)
from src.model import (
    get_model, train_model, predict, evaluate_model,
    get_feature_importance, forecast_future, compare_models
)
from src.visualization import (
    plot_time_series, plot_hourly_pattern, plot_daily_pattern,
    plot_monthly_pattern, plot_predictions, plot_feature_importance,
    plot_residuals, plot_heatmap, plot_forecast,
    plot_model_comparison, plot_multi_model_predictions
)

st.set_page_config(page_title="EV Charging Demand Prediction", page_icon="⚡", layout="wide")
st.title("⚡ EV Charging Demand Prediction System")
st.markdown("Predict EV charging demand using historical data. Compare **Linear Regression** vs **Random Forest**.")

st.sidebar.header("⚙️ Configuration")

# Setup Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY", "")
api_key_input = st.sidebar.text_input("Groq API Key", value=groq_api_key, type="password", help="Required for Milestone 2: Agentic Planning")
if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

data_source = st.sidebar.radio("Select Data Source", ["Sample Data", "Local File Path", "Upload Data"])


def load_sample_data():
    sample_path = "data/volume.csv"
    if os.path.exists(sample_path):
        return load_volume_data(sample_path)
    st.error("Sample data not found. Please upload your own data.")
    return None


def load_local_file(filepath):
    if os.path.exists(filepath):
        return load_volume_data(filepath)
    st.error(f"File not found: {filepath}")
    return None


def run_analysis(df):
    # 1. Data Overview
    st.header("📊 1. Data Overview")
    df_processed = prepare_features(df)
    stats = get_summary_stats(df_processed)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{stats['total_records']:,}")
    col2.metric("Date Range", f"{stats['date_range_start']} to {stats['date_range_end']}")
    col3.metric("Avg Volume", f"{stats['avg_volume']:.2f} kWh")
    col4.metric("Max Volume", f"{stats['max_volume']:.2f} kWh")

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    # 2. Demand Patterns
    st.header("📈 2. Demand Patterns")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time Series", "Hourly", "Daily", "Monthly", "Heatmap"])

    with tab1:
        st.plotly_chart(plot_time_series(df_processed), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_hourly_pattern(df_processed), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_daily_pattern(df_processed), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_monthly_pattern(df_processed), use_container_width=True)
    with tab5:
        st.plotly_chart(plot_heatmap(df_processed), use_container_width=True)

    # 3. Peak Demand
    st.header("🔥 3. Peak Demand Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Peak Hours")
        for hour, volume in get_peak_hours(df_processed).items():
            st.write(f"🕐 Hour {hour}:00 — **{volume:.2f} kWh** avg")

    with col2:
        st.subheader("Peak Days")
        for day, volume in get_peak_days(df_processed).items():
            st.write(f"📅 {day} — **{volume:.2f} kWh** avg")

    # 4. Model Training & Comparison
    st.header("🤖 4. Model Training & Comparison")
    st.markdown("Training **Linear Regression** and **Random Forest** on the same data to compare accuracy.")

    train_ratio = st.slider("Training Data Ratio", 0.6, 0.9, 0.8)

    if st.button("🚀 Train & Compare Models"):
        with st.spinner("Training models..."):
            feature_cols = get_feature_columns()
            train_df, test_df = split_data(df_processed, train_ratio)

            X_train = train_df[feature_cols].values
            y_train = train_df['total_volume'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['total_volume'].values

            comparison = compare_models(X_train, y_train, X_test, y_test)

            # save to session state
            st.session_state['comparison'] = comparison
            st.session_state['test_df'] = test_df
            st.session_state['y_test'] = y_test
            st.session_state['feature_cols'] = feature_cols
            st.session_state['df_processed'] = df_processed

        st.success("✅ Both models trained!")

        # comparison table
        st.subheader("📋 Metrics Comparison")
        rows = []
        for name, res in comparison.items():
            row = {'Model': name}
            row.update({f"Train {k}": v for k, v in res['train_metrics'].items()})
            row.update({f"Test {k}": v for k, v in res['test_metrics'].items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # pick best model
        best = min(comparison.keys(), key=lambda m: comparison[m]['test_metrics']['RMSE'])
        st.info(f"🏆 **Best Model: {best}** (lowest test RMSE)")
        st.session_state['model'] = comparison[best]['model']
        st.session_state['best_model_name'] = best

        # comparison bar charts
        st.subheader("📊 Visual Comparison")
        c1, c2, c3 = st.tabs(["RMSE", "MAE", "R² Score"])
        with c1:
            st.plotly_chart(plot_model_comparison(comparison, 'RMSE'), use_container_width=True)
        with c2:
            st.plotly_chart(plot_model_comparison(comparison, 'MAE'), use_container_width=True)
        with c3:
            st.plotly_chart(plot_model_comparison(comparison, 'R2 Score'), use_container_width=True)

        # all predictions on one chart
        st.subheader("🔍 Predictions vs Actual")
        st.plotly_chart(plot_multi_model_predictions(y_test, comparison, test_df['time'].values), use_container_width=True)

        # detailed per-model results
        for name, res in comparison.items():
            with st.expander(f"📌 {name} — Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Metrics**")
                    for k, v in res['train_metrics'].items():
                        st.metric(k, v)
                with col2:
                    st.write("**Testing Metrics**")
                    for k, v in res['test_metrics'].items():
                        st.metric(k, v)

                importance = get_feature_importance(res['model'], feature_cols)
                st.plotly_chart(plot_feature_importance(importance), use_container_width=True)
                st.plotly_chart(plot_residuals(y_test, res['predictions']), use_container_width=True)

    # 5. Forecast
    st.header("🔮 5. Future Demand Forecast")

    if 'model' in st.session_state:
        st.markdown(f"Using **{st.session_state.get('best_model_name', '')}** for forecasting.")
        forecast_hours = st.slider("Forecast Hours Ahead", 1, 48, 24)

        if st.button("📈 Generate Forecast"):
            model = st.session_state['model']
            feature_cols = st.session_state['feature_cols']
            df_processed = st.session_state['df_processed']

            last_row = df_processed.iloc[-1].copy()
            forecast = forecast_future(model, last_row, feature_cols, steps=forecast_hours)

            last_time = df_processed['time'].iloc[-1]
            forecast_times = [last_time + timedelta(hours=i + 1) for i in range(forecast_hours)]

            st.subheader("Forecasted Demand")
            hist_times = df_processed['time'].tail(100).values
            hist_vals = df_processed['total_volume'].tail(100).values
            st.plotly_chart(plot_forecast(hist_times, hist_vals, forecast_times, forecast), use_container_width=True)

            forecast_df = pd.DataFrame({
                'Time': forecast_times,
                'Predicted Volume (kWh)': [round(v, 2) for v in forecast]
            })
            st.dataframe(forecast_df)

            csv = forecast_df.to_csv(index=False)
            st.download_button("⬇️ Download Forecast CSV", csv, "ev_forecast.csv", "text/csv")
    else:
        st.info("👆 Train the models first to generate forecasts.")

    # 6. Agentic Infrastructure Planning
    st.header("🤖 6. AI Infrastructure Planning Assistant (Milestone 2)")
    st.markdown("Use our LLM-powered agent to reason about demand patterns and generate infrastructure expansion recommendations.")
    
    if st.button("🧠 Generate Planning Report (Uses LangGraph + Groq)"):
        with st.spinner("Agent is analyzing data and retrieving guidelines..."):
            stats = get_summary_stats(df_processed)
            peak_hours = get_peak_hours(df_processed)
            peak_days = get_peak_days(df_processed)
            
            # Predict dict logic
            predictions = {}
            if 'model' in st.session_state and 'df_processed' in st.session_state:
                predictions['status'] = 'Forecast available'
            
            from src.agent import run_agent
            report = run_agent(predictions, peak_hours, peak_days, stats)
            st.session_state['agent_report'] = report
            
    if 'agent_report' in st.session_state:
        report = st.session_state['agent_report']
        st.success("✅ Report Generated!")
        
        st.subheader("📝 Charging Demand Summary")
        st.info(report.get("demand_summary", ""))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 High-Load Analysis")
            st.write(report.get("high_load_analysis", ""))
        with col2:
            st.subheader("🕵️ Review Feedback")
            st.write(report.get("review_status", ""))
        
        st.subheader("🏗️ Infrastructure Expansion Recommendations")
        st.write(report.get("infrastructure_recommendations", ""))
        
        st.subheader("📅 Scheduling & Load-Balancing Insights")
        st.write(report.get("scheduling_insights", ""))
        
        with st.expander("📚 Supporting References (RAG)"):
            st.write(report.get("references", ""))
            
        if report.get("data_warnings"):
            st.warning(f"Data Warning: {report.get('data_warnings')}")
            
        from src.pdf_export import generate_planning_report
        import tempfile
        import os
        
        pdf_path = os.path.join(tempfile.gettempdir(), "ev_planning_report.pdf")
        generate_planning_report(report, pdf_path)
        
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="📥 Download Infrastructure Planning Report (PDF)",
            data=pdf_bytes,
            file_name="ev_planning_report.pdf",
            mime="application/pdf"
        )

# route to the right data source
def process_data_frame(df):
    if 'time' not in df.columns:
        # Try to find a date/time column automatically
        time_col = None
        for col in df.columns:
            try:
                # Test first 5 rows to see if it parses as datetime
                pd.to_datetime(df[col].dropna().astype(str).head())
                time_col = col
                break
            except Exception:
                continue
                
        if time_col:
            st.success(f"Auto-detected time column: '{time_col}'")
            df = df.rename(columns={time_col: 'time'})
        else:
            st.error("Could not detect any time/date column in the CSV. Please provide valid timestamps.")
            return
            
    # Make sure 'time' is the first column
    cols = ['time'] + [c for c in df.columns if c != 'time']
    df = df[cols]
    
    df['time'] = pd.to_datetime(df['time'])
    run_analysis(df)
    
    # 7. Q&A Chat Interface
    st.markdown("---")
    st.header("💬 7. Ask the AI About Your Data")
    st.markdown("Ask specific questions about the insights, peaks, or infrastructure recommendations.")
    
    user_q = st.text_input("Drop your question here:")
    if st.button("Ask AI"):
        from src.llm import ask_llm
        from src.preprocessing import get_summary_stats, get_peak_hours
        
        ctx = "No report generated yet."
        if 'agent_report' in st.session_state:
            ctx = str(st.session_state['agent_report'])
        else:
            stats = get_summary_stats(df)
            peaks = get_peak_hours(df)
            ctx = f"Basic Stats: {stats}\nPeak Hours: {peaks}"
            
        prompt = f"You are an EV planner. Context from data: {ctx}\n\nUser asks: {user_q}\nAnswer specifically and concisely."
        with st.spinner("AI is thinking..."):
            ans = ask_llm(prompt)
            st.info(ans)


if data_source == "Sample Data":
    df = load_sample_data()
    if df is not None:
        process_data_frame(df)
elif data_source == "Local File Path":
    st.sidebar.markdown("**Tip:** Enter the path to your CSV file below.")
    local_path = st.sidebar.text_input("File Path", value="data/volume.csv")
    if st.sidebar.button("Load File"):
        df = load_local_file(local_path)
        if df is not None:
            process_data_frame(df)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        process_data_frame(df)
    else:
        st.info("📂 Upload a CSV file with EV charging data.")
        st.markdown("""
        **Expected CSV Format:**
        - Even if you don't name it exactly `time`, the AI will auto-detect your timestamp column.
        - Other columns should be zone IDs / station volumes.
        """)
