from flask import Blueprint, render_template, Response, stream_with_context
import pandas as pd
import os
import json
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet # type: ignore
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
import markdown
from markupsafe import Markup
import time

demo_bp = Blueprint('demo', __name__)

# Load summarizer once for efficiency
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

DATA_PATH = os.path.join("static", "data", "demo_dataset.csv")

# Load dataset once
demo_data = pd.read_csv(DATA_PATH)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_gemini_report(stats, highlights, sample_data):
    prompt = (
        "You are a data analyst. Write a detailed, insightful report about the following dataset in easy to understand language. Try to be brief."
        "Include trends, anomalies, and business recommendations. Just generate the report, without any additional commentary like [Your Name/Data Analysis Team] **Date:** [Current Date] and all.\n"
        f"Stats: {stats}\n"
        f"Highlights: {highlights}\n"
        f"Sample Data: {sample_data[:5]}\n"
        "Report:"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

def stream_gemini_report(stats, highlights, sample_data):
    prompt = (
        "You are a data analyst. Write a detailed, insightful report about the following dataset in easy to understand language. Try to be brief."
        "Include trends, anomalies, and business recommendations. Just generate the report, without any additional commentary like [Your Name/Data Analysis Team] **Date:** [Current Date] and all.\n"
        f"Stats: {stats}\n"
        f"Highlights: {highlights}\n"
        f"Sample Data: {sample_data[:5]}\n"
        "Report:"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt, stream=True)
    
    for chunk in response:
        if hasattr(chunk, 'text'):
            yield chunk.text

@demo_bp.route('/demo/report')
def demo_report():
    df = pd.read_csv("static/data/demo_dataset.csv")
    summary_points = [
        f"Total sales: {df['sales'].sum()}",
        f"Average revenue: {df['revenue'].mean():.2f}",
        f"Customer complaints: {df['customer_complaints'].sum()}",
        f"Total marketing spend: {df['marketing_spend'].sum()}",
        f"Total downtime minutes: {df['downtime_minutes'].sum()}",
        f"Products: {', '.join(df['product'].unique())}",
        f"Regions: {', '.join(df['region'].unique())}",
        f"Date range: {df['date'].min()} to {df['date'].max()}"
    ]
    # Just render the template with stats, the report will be streamed separately
    return render_template("demo_report.html", stats=summary_points)

@demo_bp.route('/demo/report/stream')
def stream_report():
    df = pd.read_csv("static/data/demo_dataset.csv")
    summary_points = [
        f"Total sales: {df['sales'].sum()}",
        f"Average revenue: {df['revenue'].mean():.2f}",
        f"Customer complaints: {df['customer_complaints'].sum()}",
        f"Total marketing spend: {df['marketing_spend'].sum()}",
        f"Total downtime minutes: {df['downtime_minutes'].sum()}",
        f"Products: {', '.join(df['product'].unique())}",
        f"Regions: {', '.join(df['region'].unique())}",
        f"Date range: {df['date'].min()} to {df['date'].max()}"
    ]
    sample_data = df.head().to_dict(orient="records")
    
    # Use stream_with_context to maintain the request context throughout the streaming process
    return Response(stream_with_context(stream_gemini_report(summary_points, summary_points, sample_data)), 
                   mimetype='text/plain')

@demo_bp.route('/demo')
def demo_landing():
    return render_template("demo_landing.html", title="Demo")

@demo_bp.route("/demo/dashboard")
def demo_dashboard():
    # Summary statistics
    summary = demo_data.describe().reset_index().to_dict(orient="records")

    # Sample data
    sample_data = demo_data.head().to_dict(orient="records")

    # Chart data for visualizations
    # 1. Sales over time
    sales_over_time = {
        "labels": demo_data["date"].astype(str).tolist(),
        "values": [int(x) for x in demo_data["sales"].tolist()]
    }

    # 2. Revenue over time
    revenue_over_time = {
        "labels": demo_data["date"].astype(str).tolist(),
        "values": [float(x) for x in demo_data["revenue"].tolist()]
    }

    # 3. Customer complaints by product
    complaints_by_product = demo_data.groupby("product")["customer_complaints"].sum()
    complaints_product = {
        "labels": [str(x) for x in complaints_by_product.index],
        "values": [int(x) for x in complaints_by_product.values]
    }

    # 4. Downtime minutes by region
    downtime_by_region = demo_data.groupby("region")["downtime_minutes"].sum()
    downtime_region = {
        "labels": [str(x) for x in downtime_by_region.index],
        "values": [int(x) for x in downtime_by_region.values]
    }

    # 5. Marketing spend vs sales (scatter)
    marketing_vs_sales = {
        "x": [float(x) for x in demo_data["marketing_spend"].tolist()],
        "y": [int(x) for x in demo_data["sales"].tolist()]
    }

    chart_data = {
        "sales_over_time": sales_over_time,
        "revenue_over_time": revenue_over_time,
        "complaints_product": complaints_product,
        "downtime_region": downtime_region,
        "marketing_vs_sales": marketing_vs_sales
    }

    return render_template(
        "demo_dashboard.html",
        summary=summary,
        sample_data=sample_data,
        chart_data=chart_data,
    )


@demo_bp.route('/demo/anomaly')
def demo_anomaly():
    return render_template("coming_soon.html", title="Demo Anomaly Detection")

@demo_bp.route('/demo/rootcause')
def demo_rootcause():
    return render_template("coming_soon.html", title="Demo Root Cause Analysis")


@demo_bp.route("/demo/forecast")
def demo_forecast():
    forecast_days = 7
    forecast_data = {}

    # Prepare output dates
    daily = demo_data.groupby("date").agg({
        "sales": "sum",
        "revenue": "sum",
        "customer_complaints": "sum",
        "downtime_minutes": "sum"
    }).reset_index()

    last_date = pd.to_datetime(daily["date"]).iloc[-1]
    future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
    forecast_data["dates"] = future_dates

    for col in ["sales", "revenue", "customer_complaints", "downtime_minutes"]:
        # Prophet expects columns: ds (date), y (value)
        df = daily[["date", col]].rename(columns={"date": "ds", col: "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        # Get only the forecasted values for the future dates
        y_pred = forecast.tail(forecast_days)["yhat"].tolist()
        forecast_data[col] = [round(float(val), 2) for val in y_pred]

    return render_template(
        "demo_forecast.html",
        forecast_data=forecast_data,
    )

@demo_bp.route('/demo/benchmark')
def demo_benchmark():
    return render_template("coming_soon.html", title="Demo Benchmarking")

@demo_bp.route('/demo/recommend')
def demo_recommend():
    return render_template("coming_soon.html", title="Demo Recommendations")
