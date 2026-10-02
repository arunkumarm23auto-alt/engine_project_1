from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "Engine Health Percent"
ENGINE_IMAGE_URL = (
    "https://images.unsplash.com/photo-1486262715619-67b85e0b08d3"
    "?auto=format&fit=crop&w=1200&q=85"
)
ENGINE_FILES = {
    "Compression ignition (CI)": {
        "train": "ci_engine_train_data_1.csv",
        "test": "ci_engine_test_data_1.csv",
    },
    "Spark ignition (SI)": {
        "train": "si_engine_train_data_1.csv",
        "test": "si_engine_test_data_1.csv",
    },
}

st.set_page_config(
    page_title="Engine health | Predictive maintenance",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body {
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    .stApp { background: linear-gradient(180deg, #f3f9f7 0%, #eef5f4 100%); color: #17332d; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf6f4 0%, #e7f1ef 100%);
        border-right: 1px solid rgba(15,118,110,0.12);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 1.2rem !important;
    }
    h1, h2, h3, h4, p, label { color: #17332d; }
    h1 { font-size: 2.25rem !important; line-height: 1.08; }
    h2 { font-size: 1.35rem !important; }
    .brandline { color: #087f68; font-size: 0.73rem; font-weight: 800;
                 letter-spacing: 0.12em; text-transform: uppercase; }
    [data-testid="stImage"] img { border-radius: 16px; max-height: 220px; object-fit: cover; }
    .card { background: rgba(255,255,255,0.96); border: 1px solid rgba(14,84,77,0.08); border-radius: 8px;
            box-shadow: 0 8px 20px rgba(23,51,45,0.06); padding: 1rem 1.1rem; min-height: 8.5rem; }
    .header-ribbon { margin-top: 1.2rem; padding: 0.95rem 1.1rem; min-height: 8.5rem;
                     display: flex; flex-direction: column; justify-content: center; gap: 0.38rem;
                     border: 1px solid rgba(8,127,104,0.18); border-radius: 8px;
                     background: linear-gradient(115deg, #e5f3ef 0%, #ffffff 58%, #eef7f5 100%);
                     box-shadow: 0 8px 20px rgba(23,51,45,0.06); }
    .ribbon-topline { display: flex; align-items: center; justify-content: space-between; gap: 0.7rem; }
    .ribbon-live { display: inline-flex; align-items: center; gap: 0.45rem;
                   color: #087f68; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; }
    .ribbon-live::before { content: ""; width: 0.5rem; height: 0.5rem; border-radius: 50%;
                           background: #0f9b75; box-shadow: 0 0 0 3px rgba(15,155,117,0.13);
                           animation: live-pulse 1.8s ease-in-out infinite; }
    .ribbon-time { color: #17332d; font-size: 1.2rem; font-weight: 800; font-variant-numeric: tabular-nums; }
    .ribbon-count { color: #54706b; font-size: 0.78rem; }
    @keyframes live-pulse { 50% { box-shadow: 0 0 0 6px rgba(15,155,117,0.04); } }
    @media (prefers-reduced-motion: reduce) { .ribbon-live::before { animation: none; } }
    .kpi { background: linear-gradient(135deg, #f7fbfa, #edf6f5); border: 1px solid rgba(8,127,104,0.12);
           border-radius: 8px; padding: 1.1rem 1.1rem 0.9rem; box-shadow: 0 8px 20px rgba(7,94,85,0.05);
           display: flex; flex-direction: column; justify-content: space-between; min-height: 146px; }
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stMultiSelect > div,
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] .stFileUploader > section {
        background: rgba(255,255,255,0.8) !important;
        border: 1px solid rgba(15,118,110,0.18) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] {
        background: #ffffff !important;
        border: 1px solid rgba(15,118,110,0.24) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(23,51,45,0.06) !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        align-items: center !important;
        border-radius: 6px !important;
    }
    .status-pill { display: inline-block; padding: 0.38rem 0.7rem; border-radius: 999px; font-size: 0.76rem;
                   font-weight: 700; letter-spacing: 0.03em; }
    .alert-box { background: linear-gradient(135deg, #fff5f5, #ffffff); border: 1px solid rgba(207,94,80,0.2);
                 border-left: 4px solid #d04e4e; border-radius: 16px; padding: 0.9rem 1rem; }
    .section-label { color: #5e7a75; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .small-note { color: #54706b; font-size: 0.82rem; }
    div[data-testid="stVerticalBlock"] > div:has(> div > .kpi) { display: flex; }
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px solid rgba(15,118,110,0.75) !important;
        border-radius: 16px !important;
        background: linear-gradient(180deg, #ffffff 0%, #f6fbfa 100%) !important;
        box-shadow: none !important;
        min-height: 92px !important;
        padding: 0.7rem 0.9rem !important;
    }
    [data-testid="stFileUploaderDropzone"] > div:first-child {
        color: #17332d !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #0f766e, #0b5c54) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #0b665d, #0a554f) !important;
        color: #ffffff !important;
    }
    .plotly-graph-div {
        border-radius: 8px !important;
        box-shadow: 0 10px 20px rgba(8,127,104,0.04) !important;
    }
    [data-testid="stHorizontalBlock"] { align-items: stretch; }
    @media (max-width: 1000px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
        }
        [data-testid="stHorizontalBlock"]:has(.kpi) { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"]:has(.kpi) > [data-testid="stColumn"] {
            width: calc(50% - 0.5rem) !important;
            min-width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
        }
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(.kpi) > [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 0 !important;
            flex-basis: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(title, value, delta, tone="#0f766e"):
    color = {
        "healthy": "#0f766e",
        "warning": "#d97706",
        "critical": "#dc2626",
        "neutral": "#2563eb",
    }.get(tone, tone)
    st.markdown(
        f"""
        <div class="kpi">
            <div class="section-label">{title}</div>
            <div style="font-size:2.1rem; font-weight:800; color:{color}; margin-top:0.3rem;">{value}</div>
            <div class="small-note" style="margin-top:0.2rem;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def render_live_ribbon(record_count, sensor_count):
    st.markdown(
        f"""
        <div class="header-ribbon">
            <div class="ribbon-topline">
                <span class="ribbon-live">Live monitor</span>
                <span class="section-label">Local time</span>
            </div>
            <div class="ribbon-time">{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="ribbon-count">{record_count:,} records · {sensor_count} sensors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_engine_data(engine_name):
    files = ENGINE_FILES[engine_name]
    train_path = ROOT / files["train"]
    test_path = ROOT / files["test"]
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {files['train']} and {files['test']} beside app.py."
        )

    training_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    if TARGET_COLUMN not in training_data or TARGET_COLUMN not in test_data:
        raise ValueError(f"Both files must contain the target column '{TARGET_COLUMN}'.")
    features = [
        column
        for column in training_data.columns
        if column != TARGET_COLUMN
        and column in test_data.columns
        and pd.api.types.is_numeric_dtype(training_data[column])
    ]
    if not features:
        raise ValueError(f"No shared numeric sensor columns found for {engine_name}.")
    return training_data, test_data, features


@st.cache_resource(show_spinner="Training predictive models...")
def train_model(engine_name, features):
    training_data, test_data, _ = load_engine_data(engine_name)
    features = list(features)
    training_labels = pd.to_numeric(training_data[TARGET_COLUMN], errors="coerce")
    valid_training_rows = training_labels.notna()
    training_values = training_data.loc[valid_training_rows, features].apply(
        pd.to_numeric, errors="coerce"
    )

    model = HistGradientBoostingRegressor(
        max_iter=250,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(training_values, training_labels.loc[valid_training_rows])

    test_labels = pd.to_numeric(test_data[TARGET_COLUMN], errors="coerce")
    valid_test_rows = test_labels.notna()
    test_values = test_data.loc[valid_test_rows, features].apply(
        pd.to_numeric, errors="coerce"
    )
    test_predictions = np.clip(model.predict(test_values), 0, 100)

    mae = mean_absolute_error(test_labels.loc[valid_test_rows], test_predictions)
    rmse = np.sqrt(mean_squared_error(test_labels.loc[valid_test_rows], test_predictions))
    r_squared = r2_score(test_labels.loc[valid_test_rows], test_predictions)

    importance = permutation_importance(
        model,
        test_values,
        test_labels.loc[valid_test_rows],
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    ranked_sensors = sorted(
        zip(features, importance.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r_squared,
        "ranked": ranked_sensors,
        "actual": test_labels.loc[valid_test_rows].to_numpy(),
        "predicted": test_predictions,
    }


def make_gauge(title, value, color, suffix="%", minimum=0, maximum=100, scale="health"):
    if scale == "anomaly":
        steps = [
            {"range": [minimum, maximum * 0.35], "color": "#dff3eb"},
            {"range": [maximum * 0.35, maximum * 0.7], "color": "#f9eec7"},
            {"range": [maximum * 0.7, maximum], "color": "#fde8e8"},
        ]
    elif scale == "neutral":
        steps = [{"range": [minimum, maximum], "color": "#edf2f0"}]
    else:
        steps = [
            {"range": [minimum, maximum * 0.35], "color": "#fde8e8"},
            {"range": [maximum * 0.35, maximum * 0.7], "color": "#f9eec7"},
            {"range": [maximum * 0.7, maximum], "color": "#dff3eb"},
        ]

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": suffix, "font": {"size": 30, "color": "#17332d"}},
            title={"text": title, "font": {"size": 14, "color": "#52645f"}},
            gauge={
                "axis": {"range": [minimum, maximum], "tickcolor": "#87958f"},
                "bar": {"color": color, "thickness": 0.24},
                "bgcolor": "#ffffff",
                "bordercolor": "#dce5e1",
                "borderwidth": 1,
                "steps": steps,
            },
        )
    )
    figure.update_layout(
        height=205,
        margin={"l": 18, "r": 18, "t": 48, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"},
    )
    return figure


def get_condition(health_score, failure_risk):
    if health_score < 45 or failure_risk >= 65:
        return "Critical", "critical"
    if health_score < 70 or failure_risk >= 35:
        return "Warning", "warning"
    return "Healthy", "healthy"


def get_recommendation(sensor_name, health_score, risk):
    map_sensor = {
        "Soot Content Percent": "Inspect DPF regeneration cycle and soot loading.",
        "Vibration Hz": "Check bearings, mounts, and rotating imbalance.",
        "DPF Differential Pressure mbar": "Inspect filter backpressure and exhaust flow.",
        "Engine Oil Pressure bar": "Review lubrication system, oil filter, and pressure supply.",
        "Coolant Temp C": "Verify coolant flow, radiator performance, and thermostat operation.",
        "Engine Oil Temp C": "Check oil viscosity, cooling efficiency, and thermal load.",
        "Exhaust Gas Temp C": "Review combustion quality and system temperature balance.",
        "EGT Turbo": "Inspect turbo operation and boost temperature integrity.",
    }
    action = map_sensor.get(sensor_name, f"Inspect {sensor_name} and verify sensor health.")
    if health_score < 45 or risk >= 65:
        priority = "Immediate"
        interval = "1-2 weeks"
    elif health_score < 70 or risk >= 35:
        priority = "High"
        interval = "30-45 days"
    else:
        priority = "Routine"
        interval = "90 days"
    return action, priority, interval


def detect_time_column(df):
    for column in df.columns:
        name = str(column).lower()
        if "time" in name or "date" in name or "timestamp" in name:
            return column
    return None


with st.sidebar:
    st.markdown("### Controls")
    engine_name = st.radio("Engine type", list(ENGINE_FILES), index=0)
    uploaded_file = st.file_uploader(
        "Dataset",
        type=["csv", "xlsx", "xls"],
        help="Use your own engine data for live health prediction and maintenance review.",
    )

try:
    training_data, test_data, all_features = load_engine_data(engine_name)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            input_data = pd.read_excel(uploaded_file)
        else:
            input_data = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()
else:
    input_data = pd.read_csv(ROOT / ENGINE_FILES[engine_name]["test"])
    source_label = "Bundled demo dataset"

if input_data.empty:
    st.error("The uploaded dataset is empty. Please add at least one record.")
    st.stop()

available_features = [feature for feature in all_features if feature in input_data.columns]
if not available_features:
    st.error(
        "No matching sensor columns were found in the selected dataset. Please upload a file with CI/SI engine metrics."
    )
    st.stop()

sensor_df = input_data[available_features].apply(pd.to_numeric, errors="coerce")
used_features = [feature for feature in available_features if sensor_df[feature].notna().any()]
if not used_features:
    st.error("This dataset contains no numeric sensor values that can be analyzed.")
    st.stop()

train_result = train_model(engine_name, used_features)
model = train_result["model"]
mae = train_result["mae"]
rmse = train_result["rmse"]
r2 = train_result["r2"]
ranked_sensors = train_result["ranked"]
actual_values = train_result["actual"]
predicted_values = train_result["predicted"]

numeric_df = input_data[used_features].apply(pd.to_numeric, errors="coerce")
if numeric_df.empty:
    st.error("No valid sensor readings were detected.")
    st.stop()

numeric_df = numeric_df.fillna(numeric_df.median())
predict_df = numeric_df.copy()
raw_predictions = np.clip(model.predict(predict_df), 0, 100)
results = input_data.copy()
results["Estimated Health Score"] = np.round(raw_predictions, 1)
results["Estimated RUL Percent"] = np.round(np.clip(raw_predictions, 0, 100), 1)
results["Failure Risk %"] = np.clip(100 - results["Estimated Health Score"], 0, 100)

if TARGET_COLUMN in results.columns:
    results["Actual Health Score"] = pd.to_numeric(results[TARGET_COLUMN], errors="coerce")
else:
    results["Actual Health Score"] = results["Estimated Health Score"]

if detect_time_column(results):
    time_col = detect_time_column(results)
    try:
        results["timestamp"] = pd.to_datetime(results[time_col], errors="coerce")
    except Exception:
        results["timestamp"] = pd.to_datetime("2000-01-01")
else:
    results["timestamp"] = pd.date_range(start="2024-01-01", periods=len(results), freq="D")

results = results.sort_values("timestamp").reset_index(drop=True)
results["Vehicle Test ID"] = [f"VT-{idx + 1:04d}" for idx in range(len(results))]

with st.sidebar:
    parameter_filter = st.multiselect(
        "Parameters to display",
        options=used_features,
        default=used_features[:3],
    )
    if not parameter_filter:
        parameter_filter = used_features[:1]

    record_label = 0 if results.empty else 0

if not results.empty:
    results["Health State"], results["Health Level"] = zip(
        *results.apply(
            lambda row: get_condition(
                float(row["Estimated Health Score"]),
                float(row["Failure Risk %"]),
            ),
            axis=1,
        )
    )
else:
    results["Health State"] = pd.Series(dtype="object")
    results["Health Level"] = pd.Series(dtype="object")

condition_label, condition_tone = get_condition(
    float(results["Estimated Health Score"].iloc[record_label]) if not results.empty else 0,
    float(results["Failure Risk %"].iloc[record_label]) if not results.empty else 0,
)

if not results.empty:
    selected_record = results.iloc[record_label]
    selected_health = float(selected_record["Estimated Health Score"])
    selected_rul = float(selected_record["Estimated RUL Percent"])
    selected_risk = float(selected_record["Failure Risk %"])
    sensor_name = ranked_sensors[0][0] if ranked_sensors else used_features[0]
    recommendation_action, priority_level, maintenance_interval = get_recommendation(
        sensor_name,
        selected_health,
        selected_risk,
    )
else:
    selected_record = pd.Series()
    selected_health = 0
    selected_rul = 0
    selected_risk = 0
    sensor_name = used_features[0] if used_features else "Engine health"
    recommendation_action = "No active maintenance recommendation."
    priority_level = "Routine"
    maintenance_interval = "90 days"


iso_model = IsolationForest(contamination=0.1, random_state=42)
feature_matrix = results[used_features].apply(pd.to_numeric, errors="coerce").fillna(
    results[used_features].apply(pd.to_numeric, errors="coerce").median()
)
iso_model.fit(feature_matrix)
results["Anomaly Flag"] = iso_model.predict(feature_matrix) == -1
results["Anomaly Score"] = -iso_model.score_samples(feature_matrix)

alert_value = selected_health if not results.empty else 0
alert_risk = selected_risk if not results.empty else 0
if condition_tone in {"warning", "critical"}:
    st.error(
        "Maintenance Required — follow-up inspection recommended for the selected engine condition.",
    )
else:
    st.success("Engine system remains within normal operating range.")

header_left, header_right = st.columns([2.2, 1])
with header_left:
    st.markdown('<div class="brandline">POWERTRAIN INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Engine Health & Predictive Maintenance Dashboard")
    st.caption(f"Operational view · {engine_name} · Source: {source_label}")
with header_right:
    render_live_ribbon(len(results), len(used_features))

kpi_columns = st.columns(4)
with kpi_columns[0]:
    metric_card(
        "Engine Health Score",
        f"{selected_health:.1f}%",
        f"Status: {condition_label}",
        tone=condition_tone,
    )
with kpi_columns[1]:
    metric_card(
        "Estimated RUL",
        f"{selected_rul:.1f}%",
        "Projected remaining service life",
        tone=condition_tone,
    )
with kpi_columns[2]:
    metric_card(
        "Failure Risk",
        f"{selected_risk:.1f}%",
        "Risk index based on health model",
        tone="warning" if selected_risk >= 35 else "healthy",
    )
with kpi_columns[3]:
    metric_card(
        "Condition",
        condition_label,
        f"Priority: {priority_level}",
        tone=condition_tone,
    )

status_col, recommendation_col = st.columns([1.15, 1.85])
with status_col:
    st.markdown('<div class="section-label">Condition Indicator</div>', unsafe_allow_html=True)
    state_color = {"healthy": "#0f766e", "warning": "#d97706", "critical": "#dc2626"}[condition_tone]
    st.markdown(
        f"""
        <div class="card" style="margin-top:0.5rem; background: linear-gradient(135deg, #ffffff, #f3fbfa);">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
                <span class="status-pill" style="background:{state_color}20; color:{state_color};">{condition_label}</span>
            </div>
            <div class="small-note">Health score: <strong>{selected_health:.1f}%</strong></div>
            <div class="small-note">Failure risk: <strong>{selected_risk:.1f}%</strong></div>
            <div class="small-note">Maintenance interval: <strong>{maintenance_interval}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with recommendation_col:
    st.markdown('<div class="section-label">Maintenance Recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card" style="margin-top:0.5rem;">
            <div style="font-weight:800; color:#17332d; margin-bottom: 0.5rem;">Detected issue</div>
            <div class="small-note" style="margin-bottom: 0.8rem;">{recommendation_action}</div>
            <div style="display:flex; gap:0.8rem; flex-wrap:wrap;">
                <span class="status-pill" style="background:#edf5f4; color:#0f766e;">Priority: {priority_level}</span>
                <span class="status-pill" style="background:#eef3ff; color:#2563eb;">Interval: {maintenance_interval}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

trend_col, anomaly_col = st.columns([1.7, 1])
with trend_col:
    st.markdown('<div class="section-label">Trend Analysis</div>', unsafe_allow_html=True)
    if not results.empty:
        chart_df = results[parameter_filter + ["Estimated RUL Percent"]].copy()
        chart_df = chart_df.sort_values("Estimated RUL Percent").reset_index(drop=True)
        chart_fig = go.Figure()
        palette = ["#0f766e", "#2563eb", "#f59e0b", "#14b8a6", "#7c3aed", "#ef4444"]
        for idx, parameter in enumerate(parameter_filter):
            chart_fig.add_trace(
                go.Scatter(
                    x=chart_df["Estimated RUL Percent"],
                    y=chart_df[parameter],
                    mode="lines+markers",
                    name=parameter,
                    line={"color": palette[idx % len(palette)], "width": 2.4},
                    marker={"size": 5, "color": palette[idx % len(palette)]},
                    line_shape="spline",
                )
            )
        chart_fig.update_layout(
            height=365,
            legend={"orientation": "h", "y": 1.18, "x": 0},
            margin={"l": 12, "r": 12, "t": 12, "b": 12},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            xaxis={
                "title": "Estimated RUL (%)",
                "showgrid": False,
                "range": [0, 100],
            },
            yaxis={"title": "Sensor values", "gridcolor": "#eef2f2"},
            hovermode="x unified",
        )
        st.plotly_chart(chart_fig, use_container_width=True)
with anomaly_col:
    anomaly_total = int(results["Anomaly Flag"].sum())
    st.markdown('<div class="section-label">Fault / Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card" style="margin-top:0.5rem;">
            <div style="font-size:2.2rem; font-weight:800; color:#17332d;">{anomaly_total}</div>
            <div class="small-note">anomalous records flagged</div>
            <div style="margin-top:0.8rem;" class="small-note">Isolation Forest model score indicates drift from expected engine behavior.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    anomaly_bar = go.Figure(
        go.Bar(
            x=["Healthy", "Warning", "Critical"],
            y=[
                int((results["Health State"] == "Healthy").sum()),
                int((results["Health State"] == "Warning").sum()),
                int((results["Health State"] == "Critical").sum()),
            ],
            marker_color=["#0f766e", "#f59e0b", "#dc2626"],
        )
    )
    anomaly_bar.update_layout(
        height=210,
        margin={"l": 16, "r": 16, "t": 12, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis={"title": "Condition"},
        yaxis={"title": "Records"},
    )
    st.plotly_chart(anomaly_bar, use_container_width=True)

importance_col, comparison_col = st.columns([1, 1])
with importance_col:
    st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
    top_sensors = ranked_sensors[:6]
    y_vals = [item[0] for item in top_sensors][::-1]
    x_vals = [max(item[1], 0) for item in top_sensors][::-1]
    importance_fig = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h", marker_color="#087f68"))
    importance_fig.update_layout(
        height=260,
        margin={"l": 16, "r": 16, "t": 6, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis={"title": "Relative importance"},
        yaxis={"automargin": True},
    )
    st.plotly_chart(importance_fig, use_container_width=True)
with comparison_col:
    st.markdown('<div class="section-label">Actual vs Predicted Health / RUL</div>', unsafe_allow_html=True)
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=np.arange(len(actual_values)), y=actual_values, mode="lines+markers", name="Actual Health", line={"color": "#0f766e"}))
    comparison_fig.add_trace(go.Scatter(x=np.arange(len(predicted_values)), y=predicted_values, mode="lines+markers", name="Predicted Health", line={"color": "#2563eb"}))
    comparison_fig.update_layout(
        height=260,
        margin={"l": 16, "r": 16, "t": 6, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis={"title": "Record"},
        yaxis={"title": "Health (%)"},
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

performance_cols = st.columns(3)
with performance_cols[0]:
    metric_card("MAE", f"{mae:.2f}", "Mean absolute error", tone="neutral")
with performance_cols[1]:
    metric_card("RMSE", f"{rmse:.2f}", "Root mean squared error", tone="neutral")
with performance_cols[2]:
    metric_card("R²", f"{r2:.3f}", "Model fit score", tone="healthy")

sensor_status = []
for feature in used_features:
    value = float(selected_record[feature]) if feature in selected_record.index and pd.notna(selected_record[feature]) else np.nan
    feature_values = pd.to_numeric(results[feature], errors="coerce")
    q10 = feature_values.quantile(0.10)
    q25 = feature_values.quantile(0.25)
    q75 = feature_values.quantile(0.75)
    q90 = feature_values.quantile(0.90)
    if pd.isna(value):
        status_value = "Unknown"
    elif value < q10 or value > q90:
        status_value = "Abnormal"
    elif value < q25 or value > q75:
        status_value = "Warning"
    else:
        status_value = "Normal"
    sensor_status.append({"Parameter": feature, "Value": value, "Range": f"{q10:.2f} – {q90:.2f}", "Status": status_value})

sensor_status_df = pd.DataFrame(sensor_status)

st.markdown('<div class="section-label">Sensor Status Panel</div>', unsafe_allow_html=True)
status_table = sensor_status_df[["Parameter", "Value", "Range", "Status"]].copy()
status_table["Value"] = status_table["Value"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
st.dataframe(
    status_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.TextColumn(width="small"),
        "Range": st.column_config.TextColumn(width="medium"),
    },
)

st.markdown('<div class="section-label">Historical Records</div>', unsafe_allow_html=True)
summary_table = results[["Vehicle Test ID", "timestamp", "Estimated Health Score", "Estimated RUL Percent", "Failure Risk %", "Health State"]].copy()
summary_table.rename(
    columns={
        "Vehicle Test ID": "Vehicle Test ID",
        "timestamp": "Timestamp",
        "Estimated Health Score": "Health Score",
        "Estimated RUL Percent": "RUL",
        "Failure Risk %": "Risk",
        "Health State": "Maintenance Status",
    },
    inplace=True,
)
summary_table = summary_table.head(12)
summary_table["Timestamp"] = summary_table["Timestamp"].dt.strftime("%b %d %H:%M")
st.dataframe(summary_table, use_container_width=True, hide_index=True)

report_df = results[[
    "Vehicle Test ID",
    "timestamp",
    "Estimated Health Score",
    "Estimated RUL Percent",
    "Failure Risk %",
    "Health State",
    "Anomaly Flag",
    *used_features,
]].copy()
report_df.rename(columns={"timestamp": "Timestamp"}, inplace=True)
report_df["Timestamp"] = report_df["Timestamp"].dt.strftime("%b %d %H:%M")

csv_buffer = BytesIO()
report_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    report_df.to_excel(writer, index=False)
excel_buffer.seek(0)

button_col1, button_col2 = st.columns([1, 1])
with button_col1:
    st.download_button(
        label="Download Report (CSV)",
        data=csv_buffer,
        file_name="engine_health_report.csv",
        mime="text/csv",
    )
with button_col2:
    st.download_button(
        label="Export Results (Excel)",
        data=excel_buffer,
        file_name="engine_health_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption(
    f"Data coverage: {len(results)} records • Model trained on {engine_name} sensors • Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
