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

    # 6. Agentic Planner
    st.header("🧠 6. AI Agent Infrastructure Planner")
    st.markdown("Use the LangGraph AI Assistant to generate infrastructure placement and scheduling recommendations.")

    if st.button("✨ Generate AI Planning Report"):
        if 'df_processed' not in st.session_state:
            st.warning("Please train the models first so the agent has data to analyze.")
        else:
            with st.spinner("Agent is analyzing data & retrieving guidelines..."):
                from src.agent import run_agent
                
                df_processed = st.session_state['df_processed']
                stats = get_summary_stats(df_processed)
                peak_hours = get_peak_hours(df_processed)
                peak_days = get_peak_days(df_processed)
                
                report = run_agent({}, peak_hours, peak_days, stats)
                st.session_state['agent_report'] = report
                st.success("Report generated successfully!")
                
    if 'agent_report' in st.session_state:
        report = st.session_state['agent_report']
        
        if report.get("data_warnings"):
            st.warning(f"Data Warning: {report['data_warnings']}")
            
        st.subheader("📝 AI Demand Summary")
        st.write(report.get("demand_summary", ""))
        
        st.subheader("🏭 Infrastructure Recommendations")
        st.write(report.get("infrastructure_recommendations", ""))
        
        st.subheader("⏱️ Scheduling Insights")
        st.write(report.get("scheduling_insights", ""))
        
        with st.expander("📚 References & Review Status"):
            st.write("**References:**", report.get("references", ""))
            st.write("**AI Review Feedback:**", report.get("review_status", ""))
        
        # PDF EXPORT
        from src.report import generate_pdf_report
        try:
            pdf_bytes = generate_pdf_report(report)
            st.download_button(
                label="📄 Download Full Report as PDF",
                data=pdf_bytes,
                file_name="ev_planning_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {str(e)}")

# route to the right data source
if data_source == "Sample Data":
    df = load_sample_data()
    if df is not None:
        run_analysis(df)
elif data_source == "Local File Path":
    st.sidebar.markdown("**Tip:** Enter the path to your CSV file below.")
    local_path = st.sidebar.text_input("File Path", value="data/volume.csv")
    if st.sidebar.button("Load File"):
        df = load_local_file(local_path)
        if df is not None:
            run_analysis(df)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            run_analysis(df)
        else:
            st.error("CSV must contain a 'time' column.")
    else:
        st.info("📂 Upload a CSV file with EV charging data.")
        st.markdown("""
        **Expected CSV Format:**
        - First column: `time` (timestamp like YYYY-MM-DD HH:MM)
        - Other columns: zone IDs with charging volume data
        """)
