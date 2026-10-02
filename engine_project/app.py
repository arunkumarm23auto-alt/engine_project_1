from html import escape
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #111827;
        --muted: #4b5563;
        --line: #e3e8e6;
        --accent: #0f766e;
        --accent-soft: #e6f4f1;
    }

    /* ---------- Base: everything white, text black ---------- */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], .main {
        background: #ffffff !important;
        color: var(--ink);
        font-family: "Inter", "Segoe UI", "Trebuchet MS", sans-serif;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.6rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px;
    }
    h1, h2, h3, h4, p, label, li { color: var(--ink); }

    /* ---------- Top ribbon: white, black text and icons ---------- */
    [data-testid="stHeader"], header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid var(--line);
        color: #000000 !important;
    }
    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] *,
    [data-testid="stAppToolbar"] *,
    [data-testid="stStatusWidget"] *,
    [data-testid="stMainMenu"] *,
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="collapsedControl"] * {
        color: #000000 !important;
        opacity: 1 !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stToolbar"] button:hover { background: #f1f5f4 !important; }
    [data-testid="stHeader"] svg { color: #000000 !important; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.2rem !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em;
        padding-bottom: 0.6rem;
        margin-bottom: 0.4rem;
        border-bottom: 2px solid var(--accent);
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }

    /* Radio buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        background: #ffffff !important;
        border: 1px solid #cfd8d5 !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.7rem !important;
        gap: 0.2rem !important;
    }
    [data-testid="stRadio"] label p { color: #000000 !important; font-weight: 500 !important; }
    label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* Multiselect / selectbox: white background, black text */
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cfd8d5 !important;
        border-radius: 10px !important;
        color: #000000 !important;
        min-height: 2.6rem;
    }
    [data-baseweb="select"] input, [data-baseweb="select"] span { color: #000000 !important; }
    [data-baseweb="select"] svg { fill: #000000 !important; color: #000000 !important; }
    span[data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        border: 1px solid #b7ddd6 !important;
        border-radius: 8px !important;
        color: #0b3d37 !important;
    }
    span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {
        color: #0b3d37 !important; fill: #0b3d37 !important;
    }
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="menu"], ul[role="listbox"] {
        background: #ffffff !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="menu"] li, ul[role="listbox"] li {
        color: #000000 !important; background: #ffffff !important;
    }
    [data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] { background: #f1f5f4 !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1.5px dashed #9fb3ae !important;
        border-radius: 12px !important;
        min-height: 92px !important;
        padding: 0.8rem 0.9rem !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] * { color: #000000 !important; }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover { background: #0b5c54 !important; }
    [data-testid="stFileUploader"] small { color: var(--muted) !important; }

    /* ---------- Typography blocks ---------- */
    .brandline { color: var(--accent); font-size: 0.72rem; font-weight: 800;
                 letter-spacing: 0.14em; text-transform: uppercase; }
    .page-header { display: flex; justify-content: space-between; align-items: center;
                   gap: 1.5rem; margin: 0.3rem 0 1.4rem; flex-wrap: wrap; }
    .page-title { color: #000000; font-size: 2rem; font-weight: 800; line-height: 1.15;
                  margin: 0.25rem 0 0.4rem; padding: 0; letter-spacing: -0.01em; }
    .page-sub { color: var(--muted); font-size: 0.9rem; }
    .section-label { color: #374151; font-size: 0.72rem; font-weight: 700;
                     text-transform: uppercase; letter-spacing: 0.09em; }
    .section-gap { height: 1.3rem; }
    .small-note { color: var(--muted); font-size: 0.84rem; line-height: 1.45; }

    /* ---------- Cards ---------- */
    .card, .kpi {
        background: #ffffff; border: 1px solid var(--line); border-radius: 10px;
        box-shadow: 0 1px 3px rgba(17,24,39,0.06); padding: 1rem 1.15rem;
    }
    .card { flex: 1; }
    .kpi { position: relative; overflow: hidden; display: flex; flex-direction: column;
           justify-content: space-between; gap: 0.35rem; min-height: 138px; }
    .kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px;
                   background: var(--accent-line, var(--accent)); }
    .kpi-value { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }
    .updated { min-width: 270px; flex: 0 0 auto; }

    .kpi-grid { display: grid; gap: 1rem; margin-bottom: 0.4rem; }
    .kpi-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .kpi-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .info-grid { display: grid; grid-template-columns: 1fr 1.7fr; gap: 1rem; align-items: stretch; }
    .info-cell { display: flex; flex-direction: column; gap: 0.5rem; }

    .status-pill { display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px;
                   font-size: 0.76rem; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap; }
    .banner { display: flex; align-items: center; gap: 0.75rem; background: #ffffff;
              border: 1px solid var(--line); border-left: 5px solid var(--accent);
              border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 1.2rem;
              box-shadow: 0 1px 3px rgba(17,24,39,0.06); color: #000000; font-weight: 600; }
    .banner .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); flex: 0 0 auto; }

    /* Bordered chart containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(17,24,39,0.06);
        padding: 0.35rem 0.5rem 0.2rem;
    }
    [data-testid="stHorizontalBlock"] { align-items: stretch; gap: 1rem; }
    .js-plotly-plot, .plotly-graph-div { background: #ffffff !important; }

    /* ---------- HTML tables ---------- */
    .table-wrap { border: 1px solid var(--line); border-radius: 10px; overflow: auto;
                  background: #ffffff; box-shadow: 0 1px 3px rgba(17,24,39,0.06); }
    table.clean-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    .clean-table thead th { position: sticky; top: 0; background: #f7f9f8; color: #000000;
                            text-align: left; font-weight: 700; font-size: 0.74rem;
                            text-transform: uppercase; letter-spacing: 0.06em;
                            padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--line); }
    .clean-table td { padding: 0.62rem 0.9rem; color: var(--ink);
                      border-bottom: 1px solid #eef2f1; }
    .clean-table tbody tr:last-child td { border-bottom: none; }
    .clean-table tbody tr:hover td { background: #fafcfb; }

    /* ---------- Buttons ---------- */
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important; color: #000000 !important;
        border: 1px solid var(--accent) !important; border-radius: 8px !important;
        font-weight: 600 !important; width: 100%;
    }
    [data-testid="stDownloadButton"] button * { color: #000000 !important; }
    [data-testid="stDownloadButton"] button:hover { background: var(--accent-soft) !important; }

    .footer-note { color: var(--muted); font-size: 0.78rem; text-align: center;
                   margin-top: 1.6rem; padding-top: 0.9rem; border-top: 1px solid var(--line); }

    /* ---------- Responsive ---------- */
    @media (max-width: 1000px) {
        .kpi-grid.cols-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .info-grid { grid-template-columns: 1fr; }
        .page-title { font-size: 1.6rem; }
    }
    @media (max-width: 640px) {
        .kpi-grid.cols-4, .kpi-grid.cols-3 { grid-template-columns: 1fr; }
        [data-testid="stMainBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TONE_COLORS = {
    "healthy": "#0f766e",
    "warning": "#d97706",
    "critical": "#dc2626",
    "neutral": "#2563eb",
}
PILL_STYLES = {
    "Healthy": ("#0f766e", "#e6f4f1"),
    "Normal": ("#0f766e", "#e6f4f1"),
    "Warning": ("#b45309", "#fef3c7"),
    "Abnormal": ("#b91c1c", "#fee2e2"),
    "Critical": ("#b91c1c", "#fee2e2"),
    "Unknown": ("#475569", "#f1f5f9"),
}
CHART_FONT = {"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#111827", "size": 12}


def kpi_card(title, value, note, tone="healthy"):
    color = TONE_COLORS.get(tone, tone)
    return (
        f'<div class="kpi" style="--accent-line:{color};">'
        f'<div class="section-label">{escape(str(title))}</div>'
        f'<div class="kpi-value" style="color:{color};">{escape(str(value))}</div>'
        f'<div class="small-note">{escape(str(note))}</div></div>'
    )


def kpi_grid(cards):
    st.markdown(
        f'<div class="kpi-grid cols-{len(cards)}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(f'<div class="section-label">{escape(text)}</div>', unsafe_allow_html=True)


def section_gap():
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def pill(text, fg, bg):
    return f'<span class="status-pill" style="background:{bg}; color:{fg};">{escape(str(text))}</span>'


def html_table(df, pill_columns=(), max_height=None):
    head = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            text = "" if pd.isna(value) else str(value)
            if column in pill_columns and text in PILL_STYLES:
                fg, bg = PILL_STYLES[text]
                cell = pill(text, fg, bg)
            else:
                cell = escape(text)
            cells.append(f"<td>{cell}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    style = f' style="max-height:{max_height}px;"' if max_height else ""
    return (
        f'<div class="table-wrap"{style}><table class="clean-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def style_figure(figure, height, margin=None, legend_bottom=False):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=margin or {"l": 12, "r": 12, "t": 10, "b": 10},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=CHART_FONT,
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#111827"}},
        legend={
            "orientation": "h",
            "font": {"color": "#111827", "size": 12},
            "bgcolor": "rgba(255,255,255,0)",
            **({"y": -0.22, "x": 0, "yanchor": "top"} if legend_bottom else {"y": 1.12, "x": 0}),
        },
    )
    figure.update_xaxes(
        showline=True, linecolor="#9ca3af", tickfont={"color": "#111827"},
        title_font={"color": "#111827"}, gridcolor="#eef2f1", zeroline=False,
    )
    figure.update_yaxes(
        showline=True, linecolor="#9ca3af", tickfont={"color": "#111827"},
        title_font={"color": "#111827"}, gridcolor="#eef2f1", zeroline=False,
    )
    return figure


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

if condition_tone in {"warning", "critical"}:
    banner_color = TONE_COLORS[condition_tone]
    banner_text = "Maintenance required — follow-up inspection recommended for the selected engine condition."
else:
    banner_color = TONE_COLORS["healthy"]
    banner_text = "Engine system remains within normal operating range."
st.markdown(
    f'<div class="banner" style="border-left-color:{banner_color};">'
    f'<span class="dot" style="background:{banner_color};"></span>{escape(banner_text)}</div>',
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown(
    '<div class="page-header"><div>'
    '<div class="brandline">Powertrain Intelligence</div>'
    '<h1 class="page-title">Engine Health &amp; Predictive Maintenance Dashboard</h1>'
    f'<div class="page-sub">Operational view · {escape(engine_name)} · Source: {escape(str(source_label))}</div>'
    '</div>'
    '<div class="card updated"><div class="section-label">Last updated</div>'
    f'<div style="font-size:1.2rem; font-weight:700; color:#000000; margin:0.25rem 0;">{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>'
    f'<div class="small-note">Records: {len(results)} | {len(used_features)} sensors</div></div></div>',
    unsafe_allow_html=True,
)

# ---------------- KPI row ----------------
kpi_grid([
    kpi_card("Engine Health Score", f"{selected_health:.1f}%", f"Status: {condition_label}", condition_tone),
    kpi_card("Estimated RUL", f"{selected_rul:.1f}%", "Projected remaining service life", condition_tone),
    kpi_card("Failure Risk", f"{selected_risk:.1f}%", "Risk index based on health model",
             "warning" if selected_risk >= 35 else "healthy"),
    kpi_card("Condition", condition_label, f"Priority: {priority_level}", condition_tone),
])

section_gap()
state_color = TONE_COLORS[condition_tone]
st.markdown(
    '<div class="info-grid">'
    '<div class="info-cell"><div class="section-label">Condition Indicator</div>'
    '<div class="card">'
    f'<div style="margin-bottom:0.7rem;">{pill(condition_label, state_color, state_color + "20")}</div>'
    f'<div class="small-note">Health score: <strong style="color:#000;">{selected_health:.1f}%</strong></div>'
    f'<div class="small-note">Failure risk: <strong style="color:#000;">{selected_risk:.1f}%</strong></div>'
    f'<div class="small-note">Maintenance interval: <strong style="color:#000;">{escape(maintenance_interval)}</strong></div>'
    '</div></div>'
    '<div class="info-cell"><div class="section-label">Maintenance Recommendation</div>'
    '<div class="card">'
    '<div style="font-weight:800; color:#000; margin-bottom:0.45rem;">Detected issue</div>'
    f'<div class="small-note" style="margin-bottom:0.9rem;">{escape(recommendation_action)}</div>'
    '<div style="display:flex; gap:0.6rem; flex-wrap:wrap;">'
    f'{pill("Priority: " + priority_level, "#0f766e", "#e6f4f1")}'
    f'{pill("Interval: " + maintenance_interval, "#1d4ed8", "#e8efff")}'
    '</div></div></div></div>',
    unsafe_allow_html=True,
)

# ---------------- Trend + anomaly ----------------
section_gap()
trend_col, anomaly_col = st.columns([1.7, 1])
with trend_col:
    with st.container(border=True):
        section_label("Trend Analysis")
        if "timestamp" in results.columns and not results.empty:
            chart_df = results[["timestamp"] + parameter_filter + ["Estimated Health Score", "Estimated RUL Percent"]].copy()
            chart_df = chart_df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)
            if chart_df.empty:
                chart_df = results[["timestamp"] + parameter_filter + ["Estimated Health Score", "Estimated RUL Percent"]].copy()
            line_mode = "lines" if len(chart_df) > 150 else "lines+markers"
            chart_fig = go.Figure()
            palette = ["#2563eb", "#f59e0b", "#7c3aed", "#0ea5e9", "#db2777", "#64748b"]
            for idx, parameter in enumerate(parameter_filter):
                chart_fig.add_trace(
                    go.Scatter(
                        x=chart_df["timestamp"], y=chart_df[parameter], mode=line_mode, name=parameter,
                        line={"color": palette[idx % len(palette)], "width": 2},
                        marker={"size": 4, "color": palette[idx % len(palette)]},
                    )
                )
            chart_fig.add_trace(
                go.Scatter(
                    x=chart_df["timestamp"], y=chart_df["Estimated Health Score"], mode=line_mode,
                    name="Health Score", yaxis="y2",
                    line={"color": "#0f766e", "width": 2.8}, marker={"size": 4, "color": "#0f766e"},
                )
            )
            chart_fig.add_trace(
                go.Scatter(
                    x=chart_df["timestamp"], y=chart_df["Estimated RUL Percent"], mode=line_mode,
                    name="RUL", yaxis="y2",
                    line={"dash": "dot", "color": "#111827", "width": 2.4}, marker={"size": 4, "color": "#111827"},
                )
            )
            style_figure(chart_fig, 340, margin={"l": 12, "r": 12, "t": 10, "b": 10}, legend_bottom=True)
            chart_fig.update_layout(
                xaxis={"title": "Date", "showgrid": False, "tickformat": "%b %Y", "type": "date"},
                yaxis={"title": "Sensor values"},
                yaxis2={"title": "Health / RUL (%)", "overlaying": "y", "side": "right", "showgrid": False},
                hovermode="x unified",
            )
            st.plotly_chart(chart_fig, use_container_width=True)
with anomaly_col:
    with st.container(border=True):
        section_label("Fault / Anomaly Detection")
        anomaly_total = int(results["Anomaly Flag"].sum())
        anomaly_pct = 100 * anomaly_total / max(len(results), 1)
        st.markdown(
            '<div style="min-height:92px; margin:0.35rem 0 0.2rem;">'
            f'<div style="font-size:2.2rem; font-weight:800; color:#000; line-height:1.1;">{anomaly_total}'
            f'<span style="font-size:0.9rem; font-weight:600; color:#4b5563;"> &nbsp;({anomaly_pct:.1f}% of records)</span></div>'
            '<div class="small-note">anomalous records flagged by Isolation Forest — drift from expected engine behavior.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        state_counts = [int((results["Health State"] == s).sum()) for s in ("Healthy", "Warning", "Critical")]
        anomaly_bar = go.Figure(
            go.Bar(
                x=["Healthy", "Warning", "Critical"], y=state_counts, text=state_counts,
                textposition="outside", cliponaxis=False,
                marker_color=["#0f766e", "#f59e0b", "#dc2626"],
            )
        )
        style_figure(anomaly_bar, 238, margin={"l": 12, "r": 12, "t": 22, "b": 10})
        anomaly_bar.update_layout(xaxis={"title": "Condition"}, yaxis={"title": "Records"}, showlegend=False)
        st.plotly_chart(anomaly_bar, use_container_width=True)

# ---------------- Importance + actual vs predicted ----------------
section_gap()
importance_col, comparison_col = st.columns([1, 1])
with importance_col:
    with st.container(border=True):
        section_label("Feature Importance")
        top_sensors = ranked_sensors[:6]
        y_vals = [item[0] for item in top_sensors][::-1]
        x_vals = [max(item[1], 0) for item in top_sensors][::-1]
        importance_fig = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h", marker_color="#0f766e"))
        style_figure(importance_fig, 280)
        importance_fig.update_layout(xaxis={"title": "Relative importance"}, yaxis={"automargin": True}, showlegend=False)
        st.plotly_chart(importance_fig, use_container_width=True)
with comparison_col:
    with st.container(border=True):
        section_label("Actual vs Predicted Health / RUL")
        comparison_fig = go.Figure()
        comparison_fig.add_trace(go.Scatter(x=np.arange(len(actual_values)), y=actual_values, mode="lines", name="Actual Health", line={"color": "#0f766e", "width": 2}))
        comparison_fig.add_trace(go.Scatter(x=np.arange(len(predicted_values)), y=predicted_values, mode="lines", name="Predicted Health", line={"color": "#2563eb", "width": 2}))
        style_figure(comparison_fig, 280, legend_bottom=False)
        comparison_fig.update_layout(xaxis={"title": "Record"}, yaxis={"title": "Health (%)"})
        st.plotly_chart(comparison_fig, use_container_width=True)

# ---------------- Model performance ----------------
section_gap()
section_label("Model Performance")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
kpi_grid([
    kpi_card("MAE", f"{mae:.2f}", "Mean absolute error", "neutral"),
    kpi_card("RMSE", f"{rmse:.2f}", "Root mean squared error", "neutral"),
    kpi_card("R²", f"{r2:.3f}", "Model fit score", "healthy"),
])

# ---------------- Sensor status ----------------
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

status_table = pd.DataFrame(sensor_status)[["Parameter", "Value", "Range", "Status"]].copy()
status_table["Value"] = status_table["Value"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

section_gap()
section_label("Sensor Status Panel")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
st.markdown(html_table(status_table, pill_columns=("Status",), max_height=430), unsafe_allow_html=True)

# ---------------- Historical records ----------------
section_gap()
section_label("Historical Records")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
summary_table = results[["Vehicle Test ID", "timestamp", "Estimated Health Score", "Estimated RUL Percent", "Failure Risk %", "Health State"]].copy()
summary_table.rename(
    columns={
        "timestamp": "Timestamp",
        "Estimated Health Score": "Health Score",
        "Estimated RUL Percent": "RUL",
        "Failure Risk %": "Risk",
        "Health State": "Maintenance Status",
    },
    inplace=True,
)
summary_table = summary_table.head(12)
summary_table["Timestamp"] = summary_table["Timestamp"].dt.strftime("%b %d %Y %H:%M")
for numeric_col in ("Health Score", "RUL", "Risk"):
    summary_table[numeric_col] = summary_table[numeric_col].map(lambda v: f"{v:.1f}%")
st.markdown(html_table(summary_table, pill_columns=("Maintenance Status",)), unsafe_allow_html=True)

# ---------------- Export ----------------
report_df = results[[
    "Vehicle Test ID", "timestamp", "Estimated Health Score", "Estimated RUL Percent",
    "Failure Risk %", "Health State", "Anomaly Flag", *used_features,
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

section_gap()
section_label("Export")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
button_col1, button_col2, _spacer = st.columns([1, 1, 2])
with button_col1:
    st.download_button(
        label="Download Report (CSV)", data=csv_buffer,
        file_name="engine_health_report.csv", mime="text/csv",
    )
with button_col2:
    st.download_button(
        label="Export Results (Excel)", data=excel_buffer,
        file_name="engine_health_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown(
    f'<div class="footer-note">Data coverage: {len(results)} records • Model trained on {escape(engine_name)} sensors • '
    f'Last refresh: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
    unsafe_allow_html=True,
)
