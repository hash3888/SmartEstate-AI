import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import io
import html
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# =========================================================
# PAGE CONFIGURATION (must be the first Streamlit command)
# =========================================================
st.set_page_config(
    page_title="SmartEstate AI | Intelligent Real Estate",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROJECT PATHS AND FILE VALIDATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_house_prices.csv"
MODELS_DIR = BASE_DIR / "models"
EVALUATION_PATH = MODELS_DIR / "evaluation_results.csv"
HISTORY_DB = BASE_DIR / "data" / "prediction_history.db"

# =========================================================
# PREMIUM UI THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #101c3a;
    --blue: #356df3;
    --cyan: #38bdf8;
    --ink: #17233f;
    --muted: #71809b;
    --surface: #ffffff;
    --line: #e7edf6;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 8% 0%, rgba(56,189,248,.10), transparent 28%),
      radial-gradient(circle at 95% 5%, rgba(99,102,241,.09), transparent 25%),
      #f4f7fc;
    color: var(--ink);
}
.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    letter-spacing: -.035em;
}
h1 { font-weight: 800; }
h2, h3 { font-weight: 750; }
p, label, .stMarkdown { color: #42516c; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c3a 0%, #172d56 55%, #203d70 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #edf4ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: .48rem .65rem;
    border-radius: 10px;
    transition: all .2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.12);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
[data-testid="stMetric"] {
    background: rgba(255,255,255,.94);
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 17px 19px;
    box-shadow: 0 7px 24px rgba(24,45,86,.055);
    transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(24,45,86,.10);
}
[data-testid="stMetricLabel"] { color: #71809b !important; font-weight: 650; }
[data-testid="stMetricValue"] { color: #14264b !important; font-weight: 800; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border: 0;
    border-radius: 11px;
    background: linear-gradient(105deg, #356df3, #5a55e7);
    color: white;
    font-weight: 700;
    padding: .62rem 1rem;
    box-shadow: 0 6px 16px rgba(53,109,243,.18);
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(53,109,243,.28);
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    border-radius: 10px;
    border-color: #dce5f2;
}
[data-testid="stForm"], [data-testid="stExpander"], [data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 15px;
}
[data-testid="stAlert"] { border-radius: 13px; }
hr { border-color: #e3eaf4; }
.hero {
    padding: 30px 34px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(115deg, #101c3a 0%, #254c91 52%, #397df5 100%);
    box-shadow: 0 16px 36px rgba(24,57,117,.19);
    margin: 0 0 22px 0;
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "";
    position: absolute;
    width: 250px; height: 250px;
    right: -55px; top: -115px;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(255,255,255,.04), 0 0 0 58px rgba(255,255,255,.035);
}
.hero h1 { color: white !important; font-size: 2.25rem; margin: 0 0 7px 0; }
.hero p { color: #e4efff !important; margin: 0; font-size: 1rem; }
.section-card {
    background: white;
    border: 1px solid #e7edf6;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 6px 22px rgba(24,45,86,.045);
}
.sidebar-brand {
    padding: 18px 15px;
    border-radius: 16px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.13);
    margin: 8px 0 18px 0;
}
.sidebar-brand .brand-title { font-size: 1.18rem; font-weight: 800; color: white; }
.sidebar-brand .brand-sub { font-size: .78rem; color: #c8d9f6; margin-top: 4px; }
.small-muted { color: #71809b; font-size: .85rem; }
div[data-testid="stPlotlyChart"], div[data-testid="stVegaLiteChart"] {
    background: white;
    border-radius: 14px;
    padding: 8px;
}
@media (max-width: 760px) {
    .block-container { padding: 1rem .8rem 2rem .8rem; }
    .hero { padding: 23px 20px; }
    .hero h1 { font-size: 1.65rem; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE AND REPORT HELPERS
# =========================================================
def init_prediction_db():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location TEXT NOT NULL,
                area_sqft INTEGER,
                bedrooms INTEGER,
                predicted_price REAL NOT NULL,
                details_json TEXT NOT NULL
            )
        """)

def save_prediction_to_db(record):
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
            INSERT INTO prediction_history
            (created_at, location, area_sqft, bedrooms, predicted_price, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record["Date & Time"], record["Location"], record["Area (sq ft)"],
            record["Bedrooms"], record["Predicted Price (₹)"],
            json.dumps(record, ensure_ascii=False)
        ))

def load_prediction_history():
    with sqlite3.connect(HISTORY_DB) as conn:
        rows = conn.execute(
            "SELECT details_json FROM prediction_history ORDER BY id ASC"
        ).fetchall()
    return [json.loads(row[0]) for row in rows]

def clear_prediction_database():
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("DELETE FROM prediction_history")

def create_valuation_pdf(record):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=42, leftMargin=42,
        topMargin=42, bottomMargin=42, title="SmartEstate AI Property Valuation",
        author="SmartEstate AI"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=22,
        textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="ReportSub", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name="ReportSection", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#17365D"), spaceBefore=12, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="ReportNote", parent=styles["Normal"], fontSize=8,
        leading=11, textColor=colors.HexColor("#64748b")
    ))
    price = float(record.get("Predicted Price (₹)", 0))
    safe = lambda value: html.escape(str(value))
    story = [
        Paragraph("SmartEstate AI", styles["ReportTitle"]),
        Paragraph("PROPERTY VALUATION ESTIMATE", styles["ReportSub"]),
        Paragraph("Estimated Property Price", styles["ReportSection"])
    ]
    price_style = ParagraphStyle(
        "PriceValue", parent=styles["Title"], fontSize=23, leading=28,
        textColor=colors.white, alignment=TA_CENTER
    )
    price_table = Table(
        [[Paragraph(f"<b>INR {price:,.0f}</b>", price_style)]],
        colWidths=[doc.width]
    )
    price_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#17365D")),
        ("TOPPADDING", (0, 0), (-1, -1), 15),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 15)
    ]))
    story += [price_table, Paragraph("Property Details", styles["ReportSection"])]
    details = [
        ("Location", record.get("Location", "—")),
        ("Area", f'{record.get("Area (sq ft)", "—")} sq ft'),
        ("Bedrooms", record.get("Bedrooms", "—")),
        ("Bathrooms", record.get("Bathrooms", "—")),
        ("Balconies", record.get("Balconies", "—")),
        ("Floor", record.get("Floor", "—")),
        ("Total Floors", record.get("Total Floors", "—")),
        ("Parking", record.get("Parking", "—")),
        ("Furnishing", record.get("Furnishing", "—")),
        ("Transaction", record.get("Transaction", "—")),
        ("Facing", record.get("Facing", "—")),
        ("Ownership", record.get("Ownership", "—")),
        ("Overlooking", record.get("Overlooking", "—"))
    ]
    rows = [["Property Attribute", "Provided Details"]]
    rows += [[safe(k), safe(v)] for k, v in details]
    table = Table(rows, colWidths=[doc.width * .36, doc.width * .64], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d6deea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8)
    ]))
    story += [
        table, Paragraph("Report Information", styles["ReportSection"]),
        Paragraph("Generated on: " + safe(record.get("Date & Time", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))), styles["Normal"]),
        Spacer(1, 12),
        Paragraph(
            "Disclaimer: This is a machine-learning estimate based on the details supplied. "
            "It is not a verified market valuation, legal appraisal, lending assessment, "
            "or guaranteed selling price. Actual prices may vary.", styles["ReportNote"]
        ),
        Spacer(1, 8),
        Paragraph("Generated by SmartEstate AI | Final Year Major Project", styles["ReportNote"])
    ]
    doc.build(story)
    result = buffer.getvalue()
    buffer.close()
    return result

init_prediction_db()

# =========================================================
# LOAD PROJECT DATA AND MODEL
# =========================================================
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found: {DATA_PATH}")
    st.info("Please ensure data/processed/cleaned_house_prices.csv exists.")
    st.stop()

df = load_dataset(DATA_PATH)

if not EVALUATION_PATH.exists():
    st.error(f"Model evaluation file not found: {EVALUATION_PATH}")
    st.info("Please ensure models/evaluation_results.csv exists.")
    st.stop()

evaluation_df = pd.read_csv(EVALUATION_PATH)
metrics = dict(zip(evaluation_df["Metric"], evaluation_df["Value"]))

sys.path.insert(0, str(SRC_DIR))
try:
    from predict import predict_house_price
except Exception as import_error:
    st.error("Could not import the prediction function from src/predict.py.")
    st.exception(import_error)
    st.stop()

def get_options(column, fallback):
    if column in df.columns:
        values = sorted(df[column].dropna().unique().tolist(), key=lambda x: str(x))
        if values:
            return values
    return fallback

def find_dataset_column(candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]
    return None

def to_numeric_series(series):
    text = series.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False).str.replace("₹", "", regex=False)
    text = text.str.replace("Rs.", "", regex=False).str.replace("Rs", "", regex=False)
    lower = text.str.lower()
    multiplier = pd.Series(1.0, index=series.index)
    multiplier.loc[lower.str.contains(r"\b(cr|crore)\b", regex=True, na=False)] = 10000000.0
    multiplier.loc[lower.str.contains(r"\b(lac|lakh|lakhs)\b", regex=True, na=False)] = 100000.0
    text = text.str.replace(r"(?i)\b(cr|crore|lac|lakh|lakhs)\b", "", regex=True)
    numeric = pd.to_numeric(text.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")
    return numeric * multiplier

def get_recommendation_columns():
    return {
        "location": find_dataset_column(["location", "locality", "city"]),
        "price": find_dataset_column(["Amount(in rupees)", "amount in rupees", "price", "house price", "target"]),
        "area": find_dataset_column(["area", "area_sqft", "area (sq ft)", "built up area", "carpet area", "total area"]),
        "bedrooms": find_dataset_column(["Bedrooms", "bedroom", "No. of Bedrooms", "BHK"])
    }

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def page_heading(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div style="font-size:2rem">🏡</div>
      <div class="brand-title">SmartEstate AI</div>
      <div class="brand-sub">Intelligent Real Estate Platform</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "EXPLORE PLATFORM",
        [
            "Dashboard", "Price Prediction", "Property Recommendations",
            "Property Comparison", "Market Analytics",
            "EMI & Affordability Calculator", "Investment Calculator",
            "Dataset Explorer", "3D House Visualization", "About Project"
        ],
        label_visibility="visible"
    )
    st.markdown("---")
    st.caption("✨ AI-powered property intelligence")
    st.caption(f"📚 {len(df):,} dataset records")

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    hero("Find the value behind every home.", "SmartEstate AI • Intelligent House Price Prediction & Property Insights")
    st.markdown("### Your real-estate intelligence at a glance")
    a, b, c = st.columns(3)
    a.metric("AI Model", "Random Forest", "Log-price trained")
    b.metric("Dataset Size", f"{len(df):,}", "Cleaned property records")
    c.metric("Prediction Engine", "Online", "Ready to estimate")
    st.success("Machine-learning prediction engine is integrated with the application.")
    st.markdown("### 📊 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"₹{float(metrics['MAE']):,.0f}")
    m2.metric("RMSE", f"₹{float(metrics['RMSE']):,.0f}")
    m3.metric("R² Score", f"{float(metrics['R2 Score']):.4f}")
    m4.metric("MAPE", f"{float(metrics['MAPE']):.2f}%")
    st.caption("R² is the proportion of target variance explained on the evaluation split; it is not prediction accuracy.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("🏆 Feature Importance")
        feature_image = MODELS_DIR / "feature_importance.png"
        if feature_image.exists():
            st.image(str(feature_image), use_container_width=True)
        else:
            st.info("Feature importance graph not found in models/.")
    with right:
        st.subheader("🔍 SHAP Explainability")
        shap_image = MODELS_DIR / "shap_summary.png"
        if shap_image.exists():
            st.image(str(shap_image), use_container_width=True)
        else:
            st.info("SHAP summary graph not found in models/.")
    st.info("Start with **Price Prediction** in the sidebar to estimate a property's value.")

# =========================================================
# PRICE PREDICTION + HISTORY
# =========================================================
elif page == "Price Prediction":
    hero("Property Price Prediction", "Enter property characteristics and get an AI-generated estimate.")
    with st.form("property_form"):
        st.subheader("🏠 Property Information")
        c1, c2 = st.columns(2)
        with c1:
            location = st.selectbox("Property Location", get_options("location", ["Bangalore"]))
            area = st.number_input("Area (Square Feet)", 100, 50000, 1200, step=50)
            bedrooms = st.selectbox("Bedrooms", list(range(1, 11)), index=2)
            bathrooms = st.selectbox("Bathrooms", list(range(1, 9)))
        with c2:
            balcony = st.selectbox("Number of Balconies", list(range(0, 6)))
            floor_number = st.number_input("Property Floor Number", 0, 100, 3)
            total_floors = st.number_input("Total Floors in Building", 1, 100, 10)
            parking = st.selectbox("Parking Availability", list(range(0, 5)), format_func=lambda x: "No Parking" if x == 0 else str(x))
        st.markdown("---")
        st.subheader("✨ Additional Information")
        c3, c4 = st.columns(2)
        with c3:
            furnishing = st.selectbox("Furnishing Status", get_options("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"]))
            transaction = st.selectbox("Transaction Type", get_options("Transaction", ["New Property", "Resale"]))
            facing = st.selectbox("Property Facing", get_options("facing", ["East", "West", "North", "South"]))
        with c4:
            ownership = st.selectbox("Ownership", get_options("Ownership", ["Freehold", "Leasehold"]))
            overlooking = st.selectbox("Property Overlooking", get_options("overlooking", ["Garden/Park"]))
        submitted = st.form_submit_button("🚀 Predict House Price", use_container_width=True)

    if submitted:
        st.session_state["villa_area"] = area
        st.session_state["villa_bedrooms"] = bedrooms
        st.session_state["villa_floors"] = total_floors
        try:
            predicted_price = float(predict_house_price(
                area_sqft=area, bedrooms=bedrooms, bathrooms=bathrooms,
                balcony=balcony, parking_count=parking, floor_number=floor_number,
                total_floors=total_floors, location=location, furnishing=furnishing,
                transaction=transaction, facing=facing, ownership=ownership,
                overlooking=overlooking
            ))
            record = {
                "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Location": location, "Area (sq ft)": area, "Bedrooms": bedrooms,
                "Bathrooms": bathrooms, "Balconies": balcony, "Floor": floor_number,
                "Total Floors": total_floors, "Parking": parking,
                "Furnishing": furnishing, "Transaction": transaction,
                "Facing": facing, "Ownership": ownership, "Overlooking": overlooking,
                "Predicted Price (₹)": predicted_price
            }
            save_prediction_to_db(record)
            st.session_state["latest_prediction"] = record
            st.balloons()
            st.success("Prediction generated and saved to your history.")
        except Exception as error:
            st.error("Prediction failed. Please check model files and input compatibility.")
            st.exception(error)

    latest = st.session_state.get("latest_prediction")
    if latest:
        st.markdown("---")
        st.subheader("🎯 Latest Prediction")
        st.metric("Estimated Property Price", f"₹ {latest['Predicted Price (₹)']:,.0f}")
        r1, r2 = st.columns(2)
        with r1:
            st.write(f"**Location:** {latest['Location']}")
            st.write(f"**Area:** {latest['Area (sq ft)']:,} sq ft")
            st.write(f"**Bedrooms:** {latest['Bedrooms']}")
            st.write(f"**Bathrooms:** {latest['Bathrooms']}")
            st.write(f"**Balconies:** {latest['Balconies']}")
        with r2:
            st.write(f"**Floor:** {latest['Floor']} / {latest['Total Floors']}")
            st.write(f"**Parking:** {latest['Parking']}")
            st.write(f"**Furnishing:** {latest['Furnishing']}")
            st.write(f"**Transaction:** {latest['Transaction']}")
        st.download_button(
            "📄 Download Property Valuation Report (PDF)",
            create_valuation_pdf(latest),
            "SmartEstate_Property_Valuation_Report.pdf",
            "application/pdf",
            use_container_width=True
        )
        st.caption("This is an ML-based estimate, not a verified market valuation or guaranteed selling price.")

    st.markdown("---")
    st.subheader("📚 Prediction History")
    history = load_prediction_history()
    if history:
        history_df = pd.DataFrame(history)
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Predictions", f"{len(history_df):,}")
        h2.metric("Average Price", f"₹{history_df['Predicted Price (₹)'].mean():,.0f}")
        h3.metric("Highest Price", f"₹{history_df['Predicted Price (₹)'].max():,.0f}")
        h4.metric("Lowest Price", f"₹{history_df['Predicted Price (₹)'].min():,.0f}")
        st.dataframe(history_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
        st.bar_chart(history_df["Predicted Price (₹)"])
        st.download_button(
            "📥 Download Prediction History (CSV)",
            history_df.to_csv(index=False).encode("utf-8-sig"),
            "SmartEstate_Prediction_History.csv", "text/csv"
        )
        with st.expander("Danger zone: clear saved prediction history"):
            if st.button("🗑️ Clear Prediction History", type="secondary"):
                clear_prediction_database()
                st.session_state.pop("latest_prediction", None)
                st.success("Prediction history cleared.")
                st.rerun()
    else:
        st.info("No predictions saved yet. Submit the form above to build your history.")

# =========================================================
# PROPERTY RECOMMENDATIONS
# =========================================================
elif page == "Property Recommendations":
    hero("Discover matching properties", "Explore dataset properties that align with your budget and requirements.")
    st.info("Recommendations are filtered from the project dataset. They are not live listings or verified offers.")
    cols = get_recommendation_columns()
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.error("Required dataset fields not found: " + ", ".join(missing))
    else:
        data = df.copy()
        data["__price"] = to_numeric_series(data[cols["price"]])
        data["__area"] = to_numeric_series(data[cols["area"]])
        data["__beds"] = to_numeric_series(data[cols["bedrooms"]])
        data = data.dropna(subset=["__price", "__area"])
        data = data[(data["__price"] > 0) & (data["__area"] > 0)]
        if data.empty:
            st.warning("No usable price and area records.")
        else:
            locs = sorted(data[cols["location"]].dropna().astype(str).unique())
            min_price = max(1, int(data["__price"].quantile(.01)))
            max_price = max(min_price, int(data["__price"].quantile(.99)))
            default_budget = min(max_price, max(min_price, 10000000))
            min_area = max(100, int(data["__area"].quantile(.01)))
            max_area = max(min_area, int(data["__area"].quantile(.99)))
            beds = sorted(int(x) for x in data["__beds"].dropna().unique() if 0 < x <= 20)
            with st.form("recommendation_form"):
                p1, p2 = st.columns(2)
                with p1:
                    selected_location = st.selectbox("Preferred Location", ["All Locations"] + locs)
                    budget = st.number_input("Maximum Budget (₹)", 1, max_price, default_budget, step=100000)
                with p2:
                    area_pref = st.number_input("Minimum Area (sq ft)", 100, max(100, max_area), min_area, step=100)
                    bed_pref = st.selectbox("Bedrooms", ["Any"] + beds)
                find = st.form_submit_button("🔎 Find Matching Properties", use_container_width=True)
            if find:
                matches = data[data["__price"] <= budget]
                matches = matches[matches["__area"] >= area_pref]
                if selected_location != "All Locations":
                    matches = matches[matches[cols["location"]].astype(str) == selected_location]
                if bed_pref != "Any":
                    matches = matches[matches["__beds"] == float(bed_pref)]
                matches = matches.sort_values("__price", ascending=False).head(10)
                if matches.empty:
                    st.warning("No matches. Try increasing the budget or reducing area/bedroom requirements.")
                else:
                    a, b, c = st.columns(3)
                    a.metric("Matching Properties", len(matches))
                    b.metric("Average Dataset Price", f"₹{matches['__price'].mean():,.0f}")
                    c.metric("Highest Match", f"₹{matches['__price'].max():,.0f}")
                    result = pd.DataFrame({
                        "Location": matches[cols["location"]].astype(str),
                        "Area (sq ft)": matches["__area"].round().astype(int),
                        "Bedrooms": matches["__beds"].fillna(0).astype(int),
                        "Dataset Price (₹)": matches["__price"].map(lambda x: f"₹{x:,.0f}")
                    })
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.download_button("📥 Export Recommendations CSV", result.to_csv(index=False).encode("utf-8-sig"), "SmartEstate_Recommendations.csv", "text/csv")
            else:
                st.caption("Set your preferences and submit the form to see the top 10 matches.")

# =========================================================
# PROPERTY COMPARISON
# =========================================================
elif page == "Property Comparison":
    hero("Compare properties side by side", "Evaluate price, size, location and price per square foot.")
    st.info("Comparison uses historical project dataset records, not live or verified listings.")
    cols = get_recommendation_columns()
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.error("Required dataset fields not found: " + ", ".join(missing))
    else:
        data = df.copy()
        data["__price"] = to_numeric_series(data[cols["price"]])
        data["__area"] = to_numeric_series(data[cols["area"]])
        data["__beds"] = to_numeric_series(data[cols["bedrooms"]])
        data = data.dropna(subset=["__price", "__area", "__beds"])
        data = data[(data["__price"] > 0) & (data["__area"] > 0)].reset_index().rename(columns={"index": "__row"})
        if data.empty:
            st.warning("No valid property records available.")
        else:
            loc_col = cols["location"]
            data["__label"] = data.apply(lambda r: f"{r[loc_col]} | {r['__area']:,.0f} sqft | {r['__beds']:g} BHK | ₹{r['__price']:,.0f} | Row {r['__row']+1}", axis=1)
            options_df = data.sample(n=min(500, len(data)), random_state=42).sort_values("__row")
            if len(data) > 500:
                st.caption(f"Showing 500 representative choices from {len(data):,} valid records to keep the selector responsive.")
            chosen = st.multiselect("Select 2 or 3 properties", options_df["__label"].tolist(), max_selections=3, key="compare_choices")
            if len(chosen) >= 2:
                selected = options_df[options_df["__label"].isin(chosen)].set_index("__label").loc[chosen].reset_index()
                selected["Price per sq ft (₹)"] = selected["__price"] / selected["__area"]
                table = pd.DataFrame({
                    f"Property {i+1}": [
                        str(row[loc_col]), f"₹{row['__price']:,.0f}",
                        f"₹{row['Price per sq ft (₹)']:,.0f}",
                        f"{row['__area']:,.0f}", f"{row['__beds']:g}"
                    ] for i, (_, row) in enumerate(selected.iterrows())
                }, index=["Location", "Total Price (₹)", "Price per sq ft (₹)", "Area (sq ft)", "Bedrooms"])
                st.subheader("Comparison")
                st.dataframe(table, use_container_width=True)
                cc = st.columns(len(selected))
                for i, (_, row) in enumerate(selected.iterrows()):
                    cc[i].metric(f"Property {i+1}", f"₹{row['__price']:,.0f}")
                st.bar_chart(pd.Series(selected["Price per sq ft (₹)"].values, index=[f"Property {i+1}" for i in range(len(selected))]))
                st.download_button("📥 Download Comparison CSV", table.to_csv().encode("utf-8-sig"), "SmartEstate_Comparison.csv", "text/csv")
            else:
                st.caption("Select at least two properties to view a comparison.")

# =========================================================
# MARKET ANALYTICS
# =========================================================
elif page == "Market Analytics":
    hero("Real estate market analytics", "Explore price patterns across locations and property segments in the dataset.")
    st.info("Historical dataset analysis only; not a live market feed or a verified current valuation.")
    cols = get_recommendation_columns()
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.error("Required fields not found: " + ", ".join(missing))
    else:
        data = df.copy()
        data["Price"] = to_numeric_series(data[cols["price"]])
        data["Area"] = to_numeric_series(data[cols["area"]])
        data["Bedrooms"] = to_numeric_series(data[cols["bedrooms"]])
        data["Location"] = data[cols["location"]].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        data = data.dropna(subset=["Price", "Area"])
        data = data[(data["Price"] > 0) & (data["Area"] > 0)].copy()
        data["Price per sq ft"] = data["Price"] / data["Area"]
        a, b, c, d = st.columns(4)
        a.metric("Usable Properties", f"{len(data):,}")
        b.metric("Median Property Price", f"₹{data['Price'].median():,.0f}")
        c.metric("Median Price / sq ft", f"₹{data['Price per sq ft'].median():,.0f}")
        d.metric("Locations", f"{data['Location'].nunique():,}")
        st.markdown("---")
        st.subheader("📍 Location-wise analysis")
        minimum = st.selectbox("Minimum records per location", [1, 5, 10, 20], index=1)
        summary = data.groupby("Location").agg(
            Property_Count=("Price", "size"),
            Median_Price=("Price", "median"),
            Average_Price=("Price", "mean"),
            Median_Price_per_sqft=("Price per sq ft", "median")
        ).reset_index()
        summary = summary[summary["Property_Count"] >= minimum]
        if summary.empty:
            st.warning("No locations meet this record threshold.")
        else:
            l, r = st.columns(2)
            with l:
                st.markdown("**Highest median-price locations**")
                st.bar_chart(summary.nlargest(10, "Median_Price").set_index("Location")["Median_Price"])
            with r:
                st.markdown("**Lowest median-price locations**")
                st.bar_chart(summary.nsmallest(10, "Median_Price").set_index("Location")["Median_Price"])
            st.dataframe(summary.sort_values("Median_Price", ascending=False), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("📐 Area segment analysis")
        bins = [0, 500, 1000, 1500, 2000, 3000, 5000, float("inf")]
        labels = ["Up to 500", "501–1,000", "1,001–1,500", "1,501–2,000", "2,001–3,000", "3,001–5,000", "Above 5,000"]
        data["Area Range"] = pd.cut(data["Area"], bins=bins, labels=labels, include_lowest=True)
        area_summary = data.groupby("Area Range", observed=True).agg(
            Records=("Price", "size"), Median_Price=("Price", "median"),
            Median_Price_per_sqft=("Price per sq ft", "median")
        )
        x, y = st.columns(2)
        with x:
            st.bar_chart(area_summary["Median_Price"])
        with y:
            st.bar_chart(area_summary["Median_Price_per_sqft"])
        st.dataframe(area_summary, use_container_width=True)
        st.markdown("---")
        st.subheader("🛏️ Bedroom-wise analysis")
        bedroom_data = data.dropna(subset=["Bedrooms"])
        bedroom_data = bedroom_data[bedroom_data["Bedrooms"].between(1, 10)]
        if bedroom_data.empty:
            st.info("Usable bedroom values were not found.")
        else:
            bedroom_summary = bedroom_data.groupby("Bedrooms").agg(
                Records=("Price", "size"), Median_Price=("Price", "median"),
                Median_Price_per_sqft=("Price per sq ft", "median")
            )
            st.bar_chart(bedroom_summary["Median_Price"])
            st.dataframe(bedroom_summary, use_container_width=True)
        st.caption("These are segment comparisons, not a time-series price-growth chart or forecast.")

# =========================================================
# EMI AND AFFORDABILITY CALCULATOR
# =========================================================
elif page == "EMI & Affordability Calculator":
    hero("Home loan EMI calculator", "Estimate monthly repayments and test a comfortable EMI budget.")
    st.info("Illustrative calculation only. Actual rates, fees, eligibility, taxes and lender terms may differ.")
    left, right = st.columns(2)
    with left:
        property_value = st.number_input("Property Value (₹)", 100000, 1000000000, 7500000, step=100000)
        down_pct = st.slider("Down Payment (%)", 0, 90, 20)
        annual_rate = st.number_input("Annual Interest Rate (%)", 0.0, 30.0, 8.5, step=.1)
        years = st.slider("Loan Tenure (Years)", 1, 40, 20)
    with right:
        monthly_budget = st.number_input("Comfortable Monthly EMI Budget (₹)", 0, 10000000, 50000, step=5000)
        other_debt = st.number_input("Other Monthly Loan Payments (₹)", 0, 10000000, 0, step=1000)
    down_payment = property_value * down_pct / 100
    principal = property_value - down_payment
    months = years * 12
    rate = annual_rate / 1200
    if principal <= 0:
        emi = 0.0
    elif rate == 0:
        emi = principal / months
    else:
        factor = (1 + rate) ** months
        emi = principal * rate * factor / (factor - 1)
    total_payment = emi * months
    interest = max(0, total_payment - principal)
    st.markdown("---")
    st.subheader("Your loan estimate")
    a, b, c, d = st.columns(4)
    a.metric("Monthly EMI", f"₹{emi:,.0f}")
    b.metric("Loan Amount", f"₹{principal:,.0f}")
    c.metric("Down Payment", f"₹{down_payment:,.0f}")
    d.metric("Total Interest", f"₹{interest:,.0f}")
    available = monthly_budget - other_debt
    if monthly_budget > 0:
        if emi <= available:
            st.success(f"Estimated EMI fits within the entered remaining budget of ₹{available:,.0f}.")
        else:
            st.warning(f"EMI exceeds the entered remaining budget by ₹{emi-available:,.0f}.")
    st.bar_chart(pd.Series({"Principal": principal, "Interest": interest}))
    st.write(f"**Total of EMIs:** ₹{total_payment:,.0f}")
    affordable_emi = max(0, available)
    if affordable_emi <= 0:
        affordable_loan = 0.0
    elif rate == 0:
        affordable_loan = affordable_emi * months
    else:
        affordable_loan = affordable_emi * (1 - (1 + rate) ** (-months)) / rate
    st.metric("Approximate Loan Supported by Budget", f"₹{affordable_loan:,.0f}")
    st.caption("This does not assess income, credit score, lender eligibility, insurance, taxes or transaction costs.")

# =========================================================
# INVESTMENT CALCULATOR
# =========================================================
elif page == "Investment Calculator":
    hero("Property investment calculator", "Model rental yield, operating cash flow and a user-defined future-value scenario.")
    st.warning("Scenario tool only—not investment advice or a guaranteed return.")
    left, right = st.columns(2)
    with left:
        purchase = st.number_input("Purchase Price (₹)", 100000, 1000000000, 7500000, step=100000)
        rent = st.number_input("Expected Monthly Rent (₹)", 0, 10000000, 30000, step=1000)
        vacancy = st.slider("Expected Vacancy (%)", 0, 100, 5)
    with right:
        costs = st.number_input("Annual Maintenance & Operating Costs (₹)", 0, 100000000, 30000, step=5000)
        appreciation = st.slider("Assumed Annual Value Change (%)", -10.0, 20.0, 5.0, step=.5)
        years = st.slider("Projection Period (Years)", 1, 30, 5)
    potential_rent = rent * 12
    effective_rent = potential_rent * (1 - vacancy / 100)
    noi = effective_rent - costs
    gross_yield = potential_rent / purchase * 100 if purchase else 0
    net_yield = noi / purchase * 100 if purchase else 0
    future_value = purchase * (1 + appreciation / 100) ** years
    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("Gross Rental Yield", f"{gross_yield:.2f}%")
    b.metric("Net Operating Yield", f"{net_yield:.2f}%")
    c.metric("Annual Net Operating Income", f"₹{noi:,.0f}")
    d.metric(f"Scenario Value in {years} Years", f"₹{future_value:,.0f}", delta=f"₹{future_value-purchase:,.0f}")
    st.bar_chart(pd.Series({
        "Potential Rent": potential_rent, "Rent After Vacancy": effective_rent,
        "Operating Costs": costs, "Net Operating Income": noi
    }))
    st.caption("Future value is calculated from your assumed annual change rate; it is not a market forecast. Taxes and financing costs are excluded.")

# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":
    hero("Explore the property dataset", "Filter, inspect and export records from the cleaned project dataset.")
    st.info("Historical project dataset only; source records may be incomplete or outdated.")
    cols = get_recommendation_columns()
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.error("Required columns not found: " + ", ".join(missing))
    else:
        data = df.copy()
        data["__price"] = to_numeric_series(data[cols["price"]])
        data["__area"] = to_numeric_series(data[cols["area"]])
        data["__beds"] = to_numeric_series(data[cols["bedrooms"]])
        data["__location"] = data[cols["location"]].fillna("Unknown").astype(str)
        prices = data["__price"].dropna()
        areas = data["__area"].dropna()
        a, b, c, d = st.columns(4)
        a.metric("Dataset Rows", f"{len(data):,}")
        b.metric("Valid Price Records", f"{len(prices):,}")
        c.metric("Valid Area Records", f"{len(areas):,}")
        d.metric("Unique Locations", f"{data['__location'].nunique():,}")
        if prices.empty or areas.empty:
            st.warning("Numeric price or area values are unavailable.")
        else:
            locs = sorted(data["__location"].unique())
            min_price, max_price = int(prices.min()), max(1, int(prices.max()))
            min_area, max_area = int(areas.min()), max(1, int(areas.max()))
            f1, f2 = st.columns(2)
            with f1:
                location_filter = st.selectbox("Location", ["All Locations"] + locs)
                pmin = st.number_input("Minimum Price (₹)", 0, max_price, min_price, step=100000)
                amin = st.number_input("Minimum Area (sq ft)", 0, max_area, min_area, step=100)
            with f2:
                pmax = st.number_input("Maximum Price (₹)", 0, max_price, max_price, step=100000)
                amax = st.number_input("Maximum Area (sq ft)", 0, max_area, max_area, step=100)
                bed_options = sorted(int(v) for v in data["__beds"].dropna().unique() if 0 <= v <= 20)
                bed_filter = st.selectbox("Bedrooms", ["Any"] + bed_options)
            if pmin > pmax or amin > amax:
                st.error("Minimum values cannot be greater than maximum values.")
            else:
                filtered = data[data["__price"].between(pmin, pmax) & data["__area"].between(amin, amax)]
                if location_filter != "All Locations":
                    filtered = filtered[filtered["__location"] == location_filter]
                if bed_filter != "Any":
                    filtered = filtered[filtered["__beds"] == float(bed_filter)]
                st.subheader(f"Filtered results: {len(filtered):,}")
                clean_view = filtered.drop(columns=["__price", "__area", "__beds", "__location"], errors="ignore")
                st.dataframe(clean_view.head(1000), use_container_width=True, hide_index=True)
                if len(clean_view) > 1000:
                    st.caption("First 1,000 rows shown for browser performance; CSV export contains every filtered row.")
                st.download_button("📥 Download all filtered rows (CSV)", clean_view.to_csv(index=False).encode("utf-8-sig"), "SmartEstate_Filtered_Dataset.csv", "text/csv")
        st.markdown("---")
        st.subheader("Dataset completeness")
        completeness = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Missing Values": df.isna().sum().values,
            "Missing (%)": (df.isna().mean().values * 100).round(2)
        }).sort_values("Missing (%)", ascending=False)
        st.dataframe(completeness, use_container_width=True, hide_index=True)

# =========================================================
# 3D HOUSE VISUALIZATION
# =========================================================
elif page == "3D House Visualization":
    hero("Interactive 3D luxury villa", "Your latest prediction details can be reflected in the 3D house model.")
    html_path = BASE_DIR / "frontend" / "3d_house.html"
    villa_area = st.session_state.get("villa_area", 1500)
    villa_bedrooms = st.session_state.get("villa_bedrooms", 2)
    villa_floors = st.session_state.get("villa_floors", 2)
    a, b, c = st.columns(3)
    a.metric("Property Area", f"{villa_area:,} sq ft")
    b.metric("Bedrooms", villa_bedrooms)
    c.metric("Floors", villa_floors)
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        sync_script = f"""
        <script>
        window.addEventListener("load", function() {{
          setTimeout(function() {{
            const bedrooms = document.getElementById("bedrooms");
            const floors = document.getElementById("floors");
            const area = document.getElementById("area");
            if (bedrooms) {{ bedrooms.value = "{villa_bedrooms}"; bedrooms.dispatchEvent(new Event("change")); }}
            if (floors) {{ floors.value = "{villa_floors}"; floors.dispatchEvent(new Event("change")); }}
            if (area) {{ area.value = "{villa_area}"; area.dispatchEvent(new Event("input")); }}
          }}, 500);
        }});
        </script>
        """
        html_content = html_content.replace("</body>", sync_script + "</body>")
        components.html(html_content, height=760, scrolling=True)
    else:
        st.error("frontend/3d_house.html was not found. Please keep your existing 3D HTML file in the frontend folder.")

# =========================================================
# ABOUT
# =========================================================
elif page == "About Project":
    hero("About SmartEstate AI", "Final Year Major Project • AI-powered real-estate analysis")
    st.markdown("""
    ### 🏡 Project Overview
    SmartEstate AI is a machine-learning-based real estate platform designed to estimate
    residential property prices and provide supporting property insights through an
    interactive web application.

    ### Core Modules
    - House price prediction using a trained Random Forest model
    - Dataset-based property recommendations and comparisons
    - Location, area and bedroom market analytics
    - Home-loan EMI and affordability calculator
    - Rental yield and investment scenario calculator
    - Filterable dataset explorer with CSV export
    - Persistent SQLite prediction history and PDF valuation reports
    - Model performance metrics, feature importance and SHAP explainability
    - Interactive Three.js 3D house visualization

    ### Technology Stack
    **Python • Pandas • Scikit-learn • Streamlit • SQLite • ReportLab • Three.js**

    ### Important Notes
    Predictions are estimates from the project's trained model, not professional
    appraisals. Dataset-based recommendations are not live listings. Calculator
    outputs depend on user-entered assumptions and are not financial advice.
    """)
    st.success("SmartEstate AI — Making property insights more accessible.")
