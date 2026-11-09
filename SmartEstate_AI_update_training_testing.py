from pathlib import Path
import shutil

app_path = Path("frontend/app.py")
if not app_path.exists():
    raise SystemExit("frontend/app.py nahi mili. Script ko SmartEstate-AI project root se run karein.")

source = app_path.read_text(encoding="utf-8")

# Add the split utility import.
if "from sklearn.model_selection import GroupShuffleSplit" not in source:
    anchor = "import streamlit.components.v1 as components"
    if anchor not in source:
        raise SystemExit("Import anchor nahi mila; app.py unchanged hai.")
    source = source.replace(anchor, anchor + "\nfrom sklearn.model_selection import GroupShuffleSplit", 1)

# Use the actual evaluation report path.
source = source.replace(
    'EVALUATION_PATH = MODELS_DIR / "evaluation_results.csv"',
    'EVALUATION_PATH = BASE_DIR / "reports" / "model_evaluation" / "training_testing_metrics.csv"'
)

old_load = '''if not EVALUATION_PATH.exists():
    st.error(f"Model evaluation file not found: {EVALUATION_PATH}")
    st.info("Please ensure models/evaluation_results.csv exists.")
    st.stop()

evaluation_df = pd.read_csv(EVALUATION_PATH)
metrics = dict(zip(evaluation_df["Metric"], evaluation_df["Value"]))
'''
new_load = '''if not EVALUATION_PATH.exists():
    st.error(f"Model evaluation file not found: {EVALUATION_PATH}")
    st.info("Please ensure reports/model_evaluation/training_testing_metrics.csv exists.")
    st.stop()

try:
    evaluation_df = pd.read_csv(EVALUATION_PATH, index_col=0)
    evaluation_df.index = evaluation_df.index.astype(str).str.strip()
    evaluation_df.columns = evaluation_df.columns.astype(str).str.strip()

    required_columns = {"MAE", "RMSE", "R2 Score", "MAPE (%)"}
    if not {"Training", "Testing"}.issubset(set(evaluation_df.index)):
        raise ValueError("CSV mein Training aur Testing rows nahi mili.")
    if not required_columns.issubset(set(evaluation_df.columns)):
        raise ValueError(
            f"Expected columns {sorted(required_columns)}; found {list(evaluation_df.columns)}"
        )

    training_metrics = evaluation_df.loc["Training"]
    testing_metrics = evaluation_df.loc["Testing"]

    split_feature_columns = [
        "area_sqft", "bedrooms", "bathrooms", "balcony", "parking_count",
        "floor_number", "total_floors", "location", "Furnishing",
        "Transaction", "facing", "Ownership", "overlooking"
    ]
    missing_columns = [c for c in split_feature_columns if c not in df.columns]
    if missing_columns or "price" not in df.columns:
        raise ValueError(
            f"Cleaned dataset columns missing: {missing_columns + ([] if 'price' in df.columns else ['price'])}"
        )

    X_split = df[split_feature_columns]
    y_split = df["price"]
    split_groups = pd.util.hash_pandas_object(X_split, index=False)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        splitter.split(X_split, y_split, groups=split_groups)
    )

    total_count = len(df)
    train_count = len(train_indices)
    test_count = len(test_indices)
    train_pct = train_count / total_count * 100 if total_count else 0
    test_pct = test_count / total_count * 100 if total_count else 0

except Exception as evaluation_error:
    st.error("Training/testing evaluation data load nahi ho paaya.")
    st.exception(evaluation_error)
    st.stop()
'''
if old_load not in source:
    raise SystemExit("Evaluation loading block match nahi hua; app.py unchanged hai.")
source = source.replace(old_load, new_load, 1)

old_dashboard = '''    st.markdown("### 📊 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"₹{float(metrics['MAE']):,.0f}")
    m2.metric("RMSE", f"₹{float(metrics['RMSE']):,.0f}")
    m3.metric("R² Score", f"{float(metrics['R2 Score']):.4f}")
    m4.metric("MAPE", f"{float(metrics['MAPE']):.2f}%")
    st.caption("R² is the proportion of target variance explained on the evaluation split; it is not prediction accuracy.")
'''
new_dashboard = '''    st.markdown("### 🧠 Dataset Training & Testing Split")
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Cleaned Records", f"{total_count:,}")
    s2.metric("Training Records", f"{train_count:,}", f"{train_pct:.2f}% of dataset")
    s3.metric("Testing Records", f"{test_count:,}", f"{test_pct:.2f}% of dataset")

    split_chart = pd.DataFrame(
        {"Records": [train_count, test_count]},
        index=["Training", "Testing"]
    )
    st.bar_chart(split_chart, y="Records", use_container_width=True)
    st.caption(
        "GroupShuffleSplit uses test_size=20% and random_state=42. "
        "Identical feature groups stay together to reduce data leakage, so the "
        "actual record ratio may differ from an exact 80:20 split."
    )

    st.markdown("### 📊 Training vs Testing Model Performance")
    comparison_df = evaluation_df.loc[
        ["Training", "Testing"], ["MAE", "RMSE", "R2 Score", "MAPE (%)"]
    ].rename(columns={"R2 Score": "R² Score"})
    st.dataframe(
        comparison_df.style.format({
            "MAE": "₹{:,.0f}",
            "RMSE": "₹{:,.0f}",
            "R² Score": "{:.4f}",
            "MAPE (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    st.markdown("### 🎯 Testing Set Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Testing MAE", f"₹{float(testing_metrics['MAE']):,.0f}")
    m2.metric("Testing RMSE", f"₹{float(testing_metrics['RMSE']):,.0f}")
    m3.metric("Testing R² Score", f"{float(testing_metrics['R2 Score']):.4f}")
    m4.metric("Testing MAPE", f"{float(testing_metrics['MAPE (%)']):.2f}%")
    st.caption(
        "R² is the proportion of target variance explained on the held-out testing split; "
        "it is not prediction accuracy. MAE shows the average absolute prediction error."
    )
'''
if old_dashboard not in source:
    raise SystemExit("Dashboard metrics block match nahi hua; app.py unchanged hai.")
source = source.replace(old_dashboard, new_dashboard, 1)

backup_path = app_path.with_name("app_before_training_testing_update.py")
shutil.copy2(app_path, backup_path)
app_path.write_text(source, encoding="utf-8")

print("SUCCESS: frontend/app.py update ho gayi.")
print(f"Backup file: {backup_path}")
print("Training/testing split, records, percentages, chart aur performance comparison add ho gaye.")
print("Run: python -m streamlit run frontend/app.py")
