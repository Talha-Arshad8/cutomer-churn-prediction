
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# New UI Styling — Midnight / Teal Analytics
# -----------------------------
st.markdown("""
<style>
:root { --ink:#172033; --muted:#65728a; --teal:#0f9f91; --line:#e3e9f1; }
.stApp { background:#f4f7fb; color:var(--ink); }
.main .block-container { max-width:1500px; padding-top:1.35rem; padding-bottom:2.5rem; }
section[data-testid="stSidebar"] { background:#111827; border-right:1px solid #263244; }
section[data-testid="stSidebar"] > div { padding:1.25rem .9rem; }
section[data-testid="stSidebar"] * { color:#e8eef8; }
.sidebar-brand { padding:15px 12px 24px; margin-bottom:18px; border-bottom:1px solid #2a3649; }
.brand-icon { display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:15px; background:linear-gradient(135deg,#0f9f91,#38c8b7); font-size:25px; box-shadow:0 8px 20px #0f9f9140; }
.brand-title { font-size:19px; font-weight:800; line-height:1.3; color:#fff; margin-top:13px; }
.brand-subtitle { font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:#9eafc7; margin-top:5px; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { background:transparent; border:1px solid transparent; border-radius:10px; padding:9px 11px; transition:all .18s ease; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#1e293b; border-color:#334155; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background:#0f9f9124; border-color:#168f84; box-shadow:inset 3px 0 #25c4b4; }
h1,h2,h3 { color:#172033; letter-spacing:-.35px; }
h1 { font-weight:800; }
h2 { font-weight:750; }
p, .stMarkdown { color:#46546a; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:18px 19px; box-shadow:0 4px 14px #17203308; }
div[data-testid="stMetricLabel"] { color:#6b7890; font-size:13px; font-weight:600; }
div[data-testid="stMetricValue"] { color:#172033; font-weight:800; }
div[data-testid="stMetricDelta"] { font-size:12px; }
div[data-testid="stDataFrame"],div[data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:white; }
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button { background:#0f9f91; color:#fff; border:1px solid #0f9f91; border-radius:9px; padding:9px 17px; font-weight:700; transition:all .18s ease; }
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover { background:#0b8278; color:#fff; border-color:#0b8278; box-shadow:0 5px 14px #0f9f9130; }
div[data-testid="stAlert"] { border-radius:11px; }
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] > div { border-radius:9px; }
.hero-card { display:flex; flex-direction:column; position:relative; overflow:hidden; background:linear-gradient(115deg,#111827 0%,#1d3544 68%,#0f766e 140%); border-radius:22px; padding:35px 38px; margin:5px 0 24px; box-shadow:0 12px 28px #1118271c; }
.hero-card:after { content:""; position:absolute; width:290px; height:290px; right:-80px; top:-110px; border-radius:50%; border:1px solid #ffffff1f; box-shadow:0 0 0 28px #ffffff08,0 0 0 58px #ffffff06; }
.hero-label { display:inline-block; align-self:flex-start; background:#0f9f9130; border:1px solid #2dd4bf60; color:#7ce9dd; padding:7px 12px; border-radius:7px; font-weight:800; font-size:10px; letter-spacing:1.3px; }
.hero-title { font-size:clamp(32px,4vw,48px); line-height:1.08; font-weight:850; color:#fff; margin:18px 0 12px; position:relative; z-index:1; }
.hero-title span { color:#55d7c8; }
.hero-description { font-size:15px; line-height:1.8; color:#c2cfdd; max-width:700px; position:relative; z-index:1; }
.feature-card { background:#fff; border:1px solid var(--line); border-radius:15px; padding:22px; min-height:185px; box-shadow:0 4px 14px #17203308; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:15px; }
div[data-testid="stPlotlyChart"], div[data-testid="stVegaLiteChart"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:10px; }
.stCaption { color:#78869b; }
[data-testid="stMainBlockContainer"] { overflow:visible; }
.feature-card h3 { margin-top:10px; color:#172033; }
.feature-card p { color:#65728a; line-height:1.65; margin-bottom:9px; }
.section-kicker { color:#0f9f91; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; font-weight:850; }
hr { border-color:#e3e9f1; }
@media(max-width:768px){ .hero-card{padding:24px}.feature-card{min-height:0}.hero-card:after{right:-170px;} }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = Path("customer_churn_prediction.csv")

MODEL_PATH = Path(__file__).parent / "best_churn_model.pkl"
FEATURES_PATH = Path(__file__).parent / "model_features.pkl"


# -----------------------------
# Load Dataset, Model and Features
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)


# -----------------------------
# Check Required Files
# -----------------------------
if not DATA_PATH.exists():
    st.error(
        "Dataset file not found. Please check the Google Drive path."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.stop()

if not FEATURES_PATH.exists():
    st.error("Model features file not found.")
    st.stop()


# -----------------------------
# Load Files
# -----------------------------
df = load_data()
model = load_model()
features = load_features()


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="brand-icon">📊</div>
    <div class="brand-title">Customer Churn</div>
    <div class="brand-subtitle">Prediction &amp; Analytics</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Customer Explorer",
        "Churn Dashboard",
        "Churn Analysis",
        "Model Comparison",
        "Threshold Analysis",
        "Churn Prediction",
        "Retention Dashboard"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-label">CUSTOMER INTELLIGENCE  /  MACHINE LEARNING</div>
        <div class="hero-title">Customer Churn<br><span>Prediction</span></div>
        <div class="hero-description">
            Explore customer behavior, measure churn risk, and use model insights
            to support smarter customer-retention decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    total_customers = len(df)
    churned_count = 0
    if "Churn" in df.columns:
        churned_count = int(
            df["Churn"].astype(str).str.strip().str.lower().eq("yes").sum()
        )
    churn_rate = (churned_count / total_customers * 100) if total_customers else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_count:,}")
    col3.metric("Model Features", f"{len(features)}")
    col4.metric("Observed Churn Rate", f"{churn_rate:.1f}%")

    st.write("")
    # At-a-glance analytics: keep both charts aligned and sized consistently.
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Churn Distribution")
        st.caption("Actual churn labels in the loaded dataset")
        if "Churn" in df.columns:
            churn_chart = (
                df["Churn"].astype(str).str.strip().str.title()
                .value_counts().reindex(["No", "Yes"], fill_value=0)
            )
            st.bar_chart(churn_chart, height=280, color="#0f9f91")
        else:
            st.info("The dataset does not contain a Churn column.")
    with chart_right:
        st.markdown("#### Customer Snapshot")
        st.caption("Dataset composition at a glance")
        snapshot = pd.Series({
            "Retained": int(df["Churn"].astype(str).str.strip().str.title().eq("No").sum()) if "Churn" in df.columns else 0,
            "Churned": int(df["Churn"].astype(str).str.strip().str.title().eq("Yes").sum()) if "Churn" in df.columns else 0,
        })
        st.bar_chart(snapshot, height=280, color="#64748b")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
        <div class="feature-card">
            <div class="section-kicker">Explore your workspace</div>
            <h3>✦ Key Features</h3>
            <p>✓ Explore customer records with search and filters</p>
            <p>✓ Analyze churn patterns across customer segments</p>
            <p>✓ Compare four machine learning classifiers</p>
            <p>✓ Test prediction thresholds and churn probabilities</p>
            <p>✓ Prioritize retention opportunities</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="feature-card">
            <div class="section-kicker">Business intelligence</div>
            <h3>↗ Turn Insights into Action</h3>
            <p>Use churn signals to identify customers who may need support,
            review their experience, and guide evidence-based retention efforts.</p>
            <p><b>Suggested workflow:</b> Explore customers → Review churn patterns → Compare models → Plan retention actions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Dataset Preview")
    st.caption("A quick look at the customer records used in this project.")
    st.dataframe(df.head(8), use_container_width=True, height=300)

    with st.expander("View Model Features"):
        st.write(list(features))


# ==================================================
# CUSTOMER EXPLORER PAGE
# ==================================================
elif page == "Customer Explorer":

    st.title("🔎 Customer Explorer")

    st.write(
        "Search, filter, and explore customer records."
    )

    filtered_df = df.copy()

    # Search
    st.subheader("Search Customers")

    search = st.text_input(
        "Search by Customer ID or any column value",
        placeholder="e.g., 7590-VHVEG"
    )

    if search.strip():
        search_mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[search_mask]

    # Churn Status Filter
    st.subheader("Filters")

    if "Churn" in df.columns:

        churn_options = sorted(
            df["Churn"].dropna().unique().tolist()
        )

        selected_churn = st.multiselect(
            "Churn Status",
            options=churn_options,
            default=churn_options
        )

        filtered_df = filtered_df[
            filtered_df["Churn"].isin(selected_churn)
        ]

    # Contract Type Filter
    if "Contract" in df.columns:

        contract_options = sorted(
            df["Contract"].dropna().unique().tolist()
        )

        selected_contract = st.multiselect(
            "Contract Type",
            options=contract_options,
            default=contract_options
        )

        filtered_df = filtered_df[
            filtered_df["Contract"].isin(selected_contract)
        ]

    # Payment Method Filter
    if "PaymentMethod" in df.columns:

        payment_options = sorted(
            df["PaymentMethod"].dropna().unique().tolist()
        )

        selected_payment = st.multiselect(
            "Payment Method",
            options=payment_options,
            default=payment_options
        )

        filtered_df = filtered_df[
            filtered_df["PaymentMethod"].isin(selected_payment)
        ]

    # Results Summary
    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Matching Customers",
        f"{len(filtered_df):,}"
    )

    col2.metric(
        "Available Columns",
        len(filtered_df.columns)
    )

    # Customer Table
    st.subheader("Customer Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # Download Filtered CSV
    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Customers (CSV)",
        data=csv_data,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )

# ==================================================
# CHURN DASHBOARD PAGE
# ==================================================
elif page == "Churn Dashboard":

    st.title("📈 Churn Dashboard")

    st.write(
        "Explore customer churn patterns, contract types, "
        "and payment methods."
    )

    if "Churn" not in df.columns:
        st.error("The dataset does not contain a Churn column.")

    else:
        # Prepare churn labels
        churn_df = df.copy()

        churn_df["Churn"] = (
            churn_df["Churn"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        total_customers = len(churn_df)

        churned_customers = int(
            (churn_df["Churn"] == "Yes").sum()
        )

        retained_customers = int(
            (churn_df["Churn"] == "No").sum()
        )

        churn_rate = (
            churned_customers / total_customers * 100
            if total_customers > 0 else 0
        )

        # Summary metrics
        st.subheader("Overall Churn Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

        col2.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

        col3.metric(
            "Retained Customers",
            f"{retained_customers:,}"
        )

        col4.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

        st.divider()

        # Churn distribution
        st.subheader("Churn Distribution")

        churn_counts = (
            churn_df["Churn"]
            .value_counts()
            .reindex(["No", "Yes"], fill_value=0)
        )

        st.bar_chart(churn_counts)

        # Churn by contract
        if "Contract" in churn_df.columns:

            st.subheader("Churn by Contract Type")

            contract_churn = pd.crosstab(
                churn_df["Contract"],
                churn_df["Churn"]
            )

            st.bar_chart(contract_churn)

            contract_rate = (
                churn_df.groupby("Contract", observed=True)["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.write("Churn Rate by Contract Type (%)")

            st.dataframe(
                contract_rate.rename("Churn Rate (%)").round(2),
                use_container_width=True
            )

        # Churn by payment method
        if "PaymentMethod" in churn_df.columns:

            st.subheader("Churn by Payment Method")

            payment_churn = pd.crosstab(
                churn_df["PaymentMethod"],
                churn_df["Churn"]
            )

            st.bar_chart(payment_churn)

            payment_rate = (
                churn_df.groupby("PaymentMethod", observed=True)["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.write("Churn Rate by Payment Method (%)")

            st.dataframe(
                payment_rate.rename("Churn Rate (%)").round(2),
                use_container_width=True
            )

        # Monthly charges comparison
        if "MonthlyCharges" in churn_df.columns:

            st.subheader("Monthly Charges by Churn Status")

            charges_summary = (
                churn_df.groupby("Churn")["MonthlyCharges"]
                .mean()
                .reindex(["No", "Yes"])
                .dropna()
            )

            st.bar_chart(charges_summary)

            st.caption(
                "This chart compares average monthly charges "
                "for retained and churned customers."
            )

        # Tenure comparison
        if "tenure" in churn_df.columns:

            st.subheader("Average Tenure by Churn Status")

            tenure_summary = (
                churn_df.groupby("Churn")["tenure"]
                .mean()
                .reindex(["No", "Yes"])
                .dropna()
            )

            st.bar_chart(tenure_summary)

        st.info(
            "Interpretation: Compare churn rates across groups "
            "to identify customer segments that may need "
            "retention efforts. These charts show associations, "
            "not proof of causation."
        )


# ==================================================
# CHURN ANALYSIS PAGE
# ==================================================
elif page == "Churn Analysis":

    st.title("🔍 Churn Analysis")

    st.write(
        "Analyze how customer characteristics relate to churn."
    )

    if "Churn" not in df.columns:
        st.error("The dataset does not contain a Churn column.")

    else:
        analysis_df = df.copy()

        analysis_df["Churn"] = (
            analysis_df["Churn"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        analysis_df = analysis_df[
            analysis_df["Churn"].isin(["Yes", "No"])
        ]

        # Choose a category to analyze
        categorical_options = [
            col for col in [
                "Contract",
                "InternetService",
                "PaymentMethod",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PaperlessBilling",
                "AutoPay",
                "Family"
            ]
            if col in analysis_df.columns
        ]

        st.subheader("Churn by Customer Group")

        if categorical_options:

            selected_column = st.selectbox(
                "Select a customer characteristic",
                categorical_options
            )

            group_counts = pd.crosstab(
                analysis_df[selected_column],
                analysis_df["Churn"]
            ).reindex(columns=["No", "Yes"], fill_value=0)

            st.write("Customer Count by Group")

            st.bar_chart(group_counts)

            # Calculate churn percentage for each group
            group_rates = (
                analysis_df.groupby(
                    selected_column,
                    dropna=False
                )["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.write("Churn Rate by Group (%)")

            st.dataframe(
                group_rates.rename("Churn Rate (%)").round(2),
                use_container_width=True
            )

        st.divider()

        # Numeric feature analysis
        numeric_options = [
            col for col in [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "TotalService"
            ]
            if col in analysis_df.columns
        ]

        st.subheader("Numeric Feature Analysis")

        if numeric_options:

            selected_numeric = st.selectbox(
                "Select a numeric feature",
                numeric_options
            )

            analysis_df[selected_numeric] = pd.to_numeric(
                analysis_df[selected_numeric],
                errors="coerce"
            )

            numeric_summary = (
                analysis_df.groupby("Churn")[selected_numeric]
                .agg(["mean", "median", "count"])
                .reindex(["No", "Yes"])
            )

            st.write(
                f"{selected_numeric}: Retained vs Churned Customers"
            )

            st.dataframe(
                numeric_summary.round(2),
                use_container_width=True
            )

            chart_data = (
                analysis_df.groupby("Churn")[selected_numeric]
                .mean()
                .reindex(["No", "Yes"])
                .dropna()
            )

            st.bar_chart(chart_data)

        st.divider()

        # Findings
        st.subheader("Key Findings")

        st.markdown(
            """
            - Compare churn rates across contract and payment groups.
            - Check whether churned customers have shorter tenure.
            - Compare average monthly charges for churned and retained customers.
            - Use these patterns to guide further investigation.
            """
        )

        st.caption(
            "These results show associations in the dataset. "
            "They do not prove that a particular feature causes churn."
        )

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("Compare the performance of the trained classification models.")

    comparison_path = Path("/content/model_comparison.csv")

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(comparison_df, use_container_width=True)

        metric = st.selectbox(
            "Select a Metric",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        )

        st.bar_chart(
            comparison_df.set_index("Model")[[metric]]
        )
    else:
        st.warning("Model comparison results are not available.")



elif page == "Threshold Analysis":
    st.title("Threshold Analysis")
    st.write(
        "Explore how changing the prediction threshold affects "
        "customer churn classification."
    )

    import joblib
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    # Load saved test data
    X_TEST_PATH = Path("/content/x_test.pkl")
    Y_TEST_PATH = Path("/content/y_test.pkl")

    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        st.warning(
            "Test data files were not found. Please save x_test and "
            "y_test from your training notebook first."
        )
        st.stop()

    x_test = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    st.success("Test data loaded successfully.")

    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    # Predict churn probabilities using the saved model
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Confusion Matrix Counts")

    col1, col2 = st.columns(2)

    col1.metric("True Negatives", int(tn))
    col2.metric("True Positives", int(tp))
    col1.metric("False Positives", int(fp))
    col2.metric("False Negatives", int(fn))

    st.subheader("Prediction Summary")

    st.write(f"Customers predicted to churn: **{int(predictions.sum())}**")
    st.write(
        f"Customers predicted not to churn: "
        f"**{int((predictions == 0).sum())}**"
    )

    st.caption(
        "A lower threshold generally identifies more potential churners "
        "but may increase false positives. A higher threshold may miss "
        "more customers who actually churn."
    )


elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write(
        "Enter customer details to estimate the probability "
        "of customer churn."
    )

    # Customer inputs
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col2:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_charges = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )
            total_charges = st.number_input(
                "Total Charges", min_value=0.0, value=840.0
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        family = int(partner == "Yes" or dependents == "Yes")

        total_service = sum([
            phone == "Yes",
            multiple_lines == "Yes",
            online_security == "Yes",
            online_backup == "Yes",
            device_protection == "Yes",
            tech_support == "Yes",
            streaming_tv == "Yes",
            streaming_movies == "Yes"
        ])

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Family": family,
            "TotalService": total_service
        }])

        try:
            # Keep the exact feature order used during training
            customer = customer[features]
            probability = model.predict_proba(customer)[0][1]
            prediction = int(probability >= 0.5)

            st.subheader("Prediction Result")
            st.metric("Churn Probability", f"{probability:.1%}")

            if prediction == 1:
                st.error("High Risk: This customer is predicted to churn.")
                st.write(
                    "Suggested action: review customer concerns and "
                    "consider an appropriate retention offer."
                )
            else:
                st.success(
                    "Lower Risk: This customer is predicted not to churn."
                )

        except Exception as e:
            st.error(f"Prediction could not be completed: {e}")




elif page == "Retention Dashboard":
    st.title("Customer Retention Dashboard")
    st.write(
        "Identify customers at risk of leaving and prioritize "
        "retention efforts."
    )

    try:
        # Prepare customer data
        retention_df = df.copy()

        # Recreate derived features used by the model
        retention_df["Family"] = (
            (retention_df["Partner"] == "Yes") |
            (retention_df["Dependents"] == "Yes")
        ).astype(int)

        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        retention_df["TotalService"] = sum(
            (retention_df[col] == "Yes").astype(int)
            for col in service_columns
            if col in retention_df.columns
        )

        # Check that all required model features exist
        missing_features = [
            col for col in features
            if col not in retention_df.columns
        ]

        if missing_features:
            st.error(f"Missing model features: {missing_features}")
            st.stop()

        # Arrange features in the exact training order
        X_retention = retention_df[features].copy()

        # Predict churn probabilities
        probabilities = model.predict_proba(X_retention)[:, 1]

        retention_df["Churn Probability"] = probabilities
        retention_df["Predicted Churn"] = (
            probabilities >= 0.50
        ).astype(int)

        retention_df["Risk Level"] = retention_df[
            "Churn Probability"
        ].apply(
            lambda p: "High Risk" if p >= 0.50 else "Lower Risk"
        )

        # Filter high-risk customers
        high_risk = retention_df[
            retention_df["Predicted Churn"] == 1
        ].copy()

        total_customers = len(retention_df)
        high_risk_count = len(high_risk)

        high_risk_rate = (
            high_risk_count / total_customers * 100
            if total_customers > 0 else 0
        )

        revenue_at_risk = pd.to_numeric(
            high_risk["MonthlyCharges"], errors="coerce"
        ).fillna(0).sum()

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("High-Risk Customers", f"{high_risk_count:,}")
        col3.metric("High-Risk Rate", f"{high_risk_rate:.1f}%")
        col4.metric(
            "Monthly Charges at Risk",
            f"${revenue_at_risk:,.2f}"
        )

        st.caption(
            "Monthly charges at risk are an estimate, not guaranteed "
            "future revenue losses."
        )

        # Risk distribution chart
        st.subheader("Customer Risk Distribution")

        risk_counts = (
            retention_df["Risk Level"]
            .value_counts()
            .reindex(["High Risk", "Lower Risk"], fill_value=0)
        )

        st.bar_chart(risk_counts)

        # Contract analysis
        st.subheader("High-Risk Customers by Contract")

        if not high_risk.empty and "Contract" in high_risk.columns:
            st.bar_chart(high_risk["Contract"].value_counts())
        else:
            st.info("No high-risk customers to display.")

        # Priority customer list
        st.subheader("Customers to Prioritize")

        display_columns = [
            col for col in [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "Churn Probability",
                "Risk Level"
            ]
            if col in high_risk.columns
        ]

        if not high_risk.empty:
            priority_list = high_risk.sort_values(
                "Churn Probability", ascending=False
            )[display_columns].head(20)

            st.dataframe(
                priority_list,
                use_container_width=True
            )

            csv_data = priority_list.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Priority Customer List",
                data=csv_data,
                file_name="priority_customers.csv",
                mime="text/csv"
            )
        else:
            st.success(
                "No customers were classified as high risk "
                "at the current 0.50 threshold."
            )

        # Retention recommendations
        st.subheader("Suggested Retention Actions")

        st.markdown(
            """
            - **Month-to-month contracts:** Consider suitable longer-term plans.
            - **Short-tenure customers:** Provide onboarding and early support.
            - **High monthly charges:** Review plan value and available alternatives.
            - **High predicted risk:** Contact customers to understand their concerns.
            """
        )

    except Exception as e:
        st.error(f"Could not generate the retention dashboard: {e}")
