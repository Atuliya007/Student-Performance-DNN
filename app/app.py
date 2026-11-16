"""Streamlit dashboard for the saved Student Performance models."""

import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "datasets" / "student-por.csv"
SPLIT_DIR = BASE_DIR / "notebooks" / "models"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "student.ipynb"

FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]
PREDICTION_INPUT_KEYS = (
    "prediction_school", "prediction_sex", "prediction_age", "prediction_address",
    "prediction_famsize", "prediction_pstatus", "prediction_medu", "prediction_fedu",
    "prediction_mjob", "prediction_fjob", "prediction_reason", "prediction_guardian",
    "prediction_traveltime", "prediction_studytime", "prediction_failures", "prediction_absences",
    "prediction_schoolsup", "prediction_famsup", "prediction_paid", "prediction_activities",
    "prediction_nursery", "prediction_higher", "prediction_internet", "prediction_romantic",
    "prediction_famrel", "prediction_freetime", "prediction_goout", "prediction_dalc",
    "prediction_walc", "prediction_health", "prediction_g1", "prediction_g2",
)
NAV_ITEMS = [
    "Prediction",
    "Model Analytics",
    "About Project",
]

st.set_page_config(page_title="Student Performance | Academic Dashboard", layout="wide")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  :root {
    --forest:#163D32; --forest-deep:#163D32; --secondary:#245747;
    --sage:#8FAF9B; --gold:#C5A46D; --teal:#245747;
    --ink:#202923; --muted:#626D66; --paper:#F7F5EF; --line:#E1E6DF;
  }
  html, body, [class*="css"], button, input, textarea, select {
    font-family:'Inter','Segoe UI',Arial,sans-serif !important;
  }
  .stApp { background:var(--paper); color:var(--ink); font-size:16px; }
  [data-testid="stAppViewContainer"] > .main .block-container {
    max-width:1480px; padding:1.2rem 2.35rem 3.5rem;
  }
  p, li, .stMarkdown { color:var(--ink); font-size:1rem; line-height:1.72; }
  strong, b { color:#14251f; font-weight:700; }
  h1,h2,h3,h4 { color:var(--forest-deep); letter-spacing:-.035em; font-weight:700 !important; }
  h1 { font-size:1.9rem !important; line-height:1.2 !important; }
  h2 { font-size:1.45rem !important; line-height:1.3 !important; margin-top:1.5rem !important; }
  h3 { font-size:1.12rem !important; line-height:1.4 !important; margin-top:1.1rem !important; }
  [data-testid="stSidebar"] { background:var(--forest-deep); border-right:1px solid #21483e; }
  [data-testid="stSidebar"] * { color:#f5f8f4 !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d8e7df !important; font-size:.95rem; }
  [data-testid="stSidebar"] [role="radiogroup"] label { font-size:.96rem; font-weight:600; line-height:1.4; padding:.2rem 0; }
  [data-testid="stSidebar"] [role="radiogroup"] label p { font-size:.96rem !important; font-weight:600 !important; }
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:rgba(220,239,229,.12); border-radius:9px; padding:.55rem .6rem; margin:.1rem 0;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size:1.22rem !important; letter-spacing:-.02em; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#d3e2da !important; }
  .hero { padding:.3rem 0 1rem; border:0; border-radius:0; color:var(--ink);
    background:transparent; box-shadow:none; margin:.25rem 0 .5rem; }
  .hero p { color:var(--muted) !important; font-size:1rem !important; font-weight:500; line-height:1.55; margin:0; }
  .note { border:1px solid #E1E6DF; border-left:5px solid var(--teal); background:#EDF2EC;
    color:#202923; font-weight:500; font-size:.98rem; line-height:1.65;
    padding:.9rem 1.1rem; border-radius:5px 11px 11px 5px; margin:.65rem 0 1.35rem; }
  [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:10px;
    padding:.8rem .95rem; box-shadow:none; height:100%; }
  [data-testid="stMetricLabel"] p { color:#626D66 !important; font-size:.9rem !important; font-weight:600 !important; }
  [data-testid="stMetricValue"] { color:var(--forest) !important; font-size:1.55rem !important; font-weight:700 !important; }
  [data-testid="stMetricDelta"] { font-weight:600 !important; }
  [data-testid="stCaptionContainer"] { color:#626D66 !important; font-size:.88rem !important; line-height:1.55; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label p {
    color:#202923 !important; font-size:.93rem !important; font-weight:600 !important;
  }
  [data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:10px;
    padding:1.1rem 1.2rem; box-shadow:none; overflow:visible; }
  [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--line) !important; border-radius:10px; }
  .topline { display:flex; align-items:center; border-bottom:1px solid var(--line); padding:.3rem 0 .8rem; margin-bottom:1rem; }
  .topline-title { color:var(--forest); font-size:1.45rem; font-weight:700; letter-spacing:-.02em; }
  [data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stSlider"] [role="slider"] { border-color:#D3DCD3; }
  [data-testid="stFormSubmitButton"] button {
    width:100%; min-height:2.8rem; padding:.65rem 1rem; border-radius:8px;
    font-size:1rem; line-height:1.25; font-weight:700; letter-spacing:0;
    transition:background-color .15s ease, border-color .15s ease;
  }
  [data-testid="stFormSubmitButton"] button[kind="primary"],
  [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] {
    background:#163D32; color:#fff; border:1px solid #163D32;
  }
  [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
  [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
    background:#245747; color:#fff; border-color:#245747;
  }
  [data-testid="stFormSubmitButton"] button[kind="secondary"],
  [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondary"] {
    background:#fff; color:#202923; border:1px solid #AEB9B0;
  }
  [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover,
  [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondary"]:hover {
    background:#F1F3EF; color:#163D32; border-color:#626D66;
  }
  [data-testid="stFormSubmitButton"] button:focus-visible {
    outline:3px solid #8FAF9B; outline-offset:2px;
  }
  [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:11px; overflow:hidden; }
  [data-testid="stDataFrame"] * { font-size:.92rem; }
  [data-testid="stAlert"] { border-radius:11px; }
  [data-testid="stAlert"] p { font-size:.96rem; font-weight:500; }
  div[data-testid="stProgress"] > div > div { background:var(--teal); }
  hr { border-color:#E1E6DF; }
  @media (max-width:700px) {
    [data-testid="stAppViewContainer"] > .main .block-container { padding:1.2rem 1rem 2rem; }
    .topline { padding-bottom:.65rem; }
    .hero { padding:.25rem 0 .75rem; }
    .topline-title { font-size:1.3rem; }
  }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading saved model assets...")
def load_assets():
    """Load saved artifacts once per Streamlit process."""
    dnn_path = MODEL_DIR / "student_performance_dnn.keras"
    prep_path = MODEL_DIR / "preprocessor.pkl"
    rf_path = MODEL_DIR / "random_forest_model.pkl"
    if not dnn_path.exists() or not prep_path.exists():
        raise FileNotFoundError("Required DNN or preprocessor artifact is missing from models/.")
    dnn = keras.models.load_model(dnn_path)
    preprocessor = joblib.load(prep_path)
    rf = joblib.load(rf_path) if rf_path.exists() else None
    return dnn, preprocessor, rf


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_data(show_spinner="Preparing evaluation charts from saved test data...")
def load_evaluation_data():
    paths = [SPLIT_DIR / name for name in ("X_test.npy", "y_test.npy")]
    if not all(path.exists() for path in paths):
        return None
    X_test = np.load(paths[0], allow_pickle=False)
    y_test = np.load(paths[1], allow_pickle=False).reshape(-1)
    dnn, _, rf = load_assets()
    dnn_pred = dnn.predict(X_test, verbose=0).reshape(-1)
    rf_pred = rf.predict(X_test).reshape(-1) if rf is not None else None
    return X_test, y_test, dnn_pred, rf_pred


def grade_category(grade):
    if grade < 10:
        return "Needs Improvement", "This estimate suggests the student may benefit from additional academic support."
    if grade < 14:
        return "Satisfactory", "This estimate indicates satisfactory progress; consistent study habits can support further growth."
    if grade < 17:
        return "Good", "This estimate indicates good academic performance and a solid understanding of course material."
    return "Excellent", "This estimate indicates excellent academic performance."


def explain_graph(representation, purpose, reading):
    st.markdown(f"**What it shows:** {representation}  \n**Why it is used:** {purpose}  \n**How to read it:** {reading}")


def metric_bundle(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse,
            "RMSE": np.sqrt(mse), "R-squared": r2_score(actual, predicted)}


def reset_prediction_form():
    """Clear only prediction inputs when the form's Reset action is clicked."""
    for key in PREDICTION_INPUT_KEYS:
        st.session_state.pop(key, None)


def page_prediction(dnn, preprocessor):
    st.markdown('<div class="hero"><p>Estimate final Portuguese course grade using the existing trained DNN.</p></div>', unsafe_allow_html=True)
    st.markdown("""<div class="note">Complete the student profile below. The saved preprocessing pipeline transforms the 32 inputs in the model's original feature order.</div>""", unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.subheader("Student and family profile")
        a, b, c = st.columns(3)
        with a:
            school = st.selectbox("School", ["GP", "MS"], key="prediction_school")
            sex = st.selectbox("Sex", ["F", "M"], key="prediction_sex")
            age = st.number_input("Age", 15, 22, 17, key="prediction_age")
            address = st.selectbox("Home address", ["U", "R"], key="prediction_address")
            famsize = st.selectbox("Family size", ["GT3", "LE3"], key="prediction_famsize")
            Pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"], key="prediction_pstatus")
            Medu = st.select_slider("Mother's education", options=list(range(5)), value=2, key="prediction_medu")
            Fedu = st.select_slider("Father's education", options=list(range(5)), value=2, key="prediction_fedu")
        with b:
            Mjob = st.selectbox("Mother's job", ["at_home", "health", "other", "services", "teacher"], key="prediction_mjob")
            Fjob = st.selectbox("Father's job", ["at_home", "health", "other", "services", "teacher"], key="prediction_fjob")
            reason = st.selectbox("Reason for choosing school", ["course", "home", "reputation", "other"], key="prediction_reason")
            guardian = st.selectbox("Primary guardian", ["father", "mother", "other"], key="prediction_guardian")
            traveltime = st.select_slider("Travel time", options=[1, 2, 3, 4], value=1, key="prediction_traveltime")
            studytime = st.select_slider("Weekly study time", options=[1, 2, 3, 4], value=2, key="prediction_studytime")
            failures = st.select_slider("Past class failures", options=[0, 1, 2, 3], value=0, key="prediction_failures")
            absences = st.number_input("Absences", min_value=0, max_value=100, value=0, key="prediction_absences")
        with c:
            schoolsup = st.selectbox("Extra educational support", ["no", "yes"], key="prediction_schoolsup")
            famsup = st.selectbox("Family educational support", ["no", "yes"], key="prediction_famsup")
            paid = st.selectbox("Extra paid classes", ["no", "yes"], key="prediction_paid")
            activities = st.selectbox("Extra activities", ["no", "yes"], key="prediction_activities")
            nursery = st.selectbox("Attended nursery school", ["yes", "no"], key="prediction_nursery")
            higher = st.selectbox("Wants higher education", ["yes", "no"], key="prediction_higher")
            internet = st.selectbox("Internet access at home", ["yes", "no"], key="prediction_internet")
            romantic = st.selectbox("In a romantic relationship", ["no", "yes"], key="prediction_romantic")
        st.subheader("Lifestyle and current academic progress")
        d, e = st.columns(2)
        with d:
            famrel = st.slider("Family relationship quality", 1, 5, 4, key="prediction_famrel")
            freetime = st.slider("Free time after school", 1, 5, 3, key="prediction_freetime")
            goout = st.slider("Going out with friends", 1, 5, 3, key="prediction_goout")
            Dalc = st.slider("Workday alcohol consumption", 1, 5, 1, key="prediction_dalc")
            Walc = st.slider("Weekend alcohol consumption", 1, 5, 1, key="prediction_walc")
            health = st.slider("Current health", 1, 5, 3, key="prediction_health")
        with e:
            G1 = st.slider("First period grade (G1)", 0, 20, 10, key="prediction_g1")
            G2 = st.slider("Second period grade (G2)", 0, 20, 10, key="prediction_g2")
            st.caption("Grades use the dataset's 0-20 scale. G1 and G2 are important intermediate grades used by the original model.")
        predict_col, reset_col = st.columns(2, gap="medium")
        with predict_col:
            submitted = st.form_submit_button(
                "Predict Output", type="primary", use_container_width=True
            )
        with reset_col:
            st.form_submit_button(
                "Reset", type="secondary", on_click=reset_prediction_form,
                use_container_width=True,
            )

    if submitted:
        student_data = dict(zip(FEATURES, [school, sex, age, address, famsize, Pstatus, Medu, Fedu,
            Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid,
            activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc,
            health, absences, G1, G2]))
        input_frame = pd.DataFrame([student_data], columns=FEATURES)
        try:
            transformed = np.asarray(preprocessor.transform(input_frame), dtype=np.float32)
            estimate = float(np.clip(dnn.predict(transformed, verbose=0).reshape(-1)[0], 0, 20))
        except Exception as exc:
            st.error(f"The saved preprocessing pipeline could not process these inputs: {exc}")
            return
        category, interpretation = grade_category(estimate)
        st.divider()
        x, y = st.columns([1, 2], gap="large")
        with x, st.container(border=True):
            st.caption("PREDICTED FINAL GRADE")
            st.metric("Grade estimate", f"{estimate:.2f} / 20")
            st.progress(int(round(estimate / 20 * 100)))
            st.caption("Grade scale: 0-20")
        with y, st.container(border=True):
            st.caption("PERFORMANCE CATEGORY")
            st.subheader(category)
            st.write(interpretation)
            st.caption("A model estimate for learning and demonstration. It is not a guaranteed academic outcome.")


@st.cache_data(show_spinner=False)
def load_notebook_figures():
    """Read the original rendered figures stored in the project notebook."""
    if not NOTEBOOK_PATH.exists():
        return {}
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    figures = {}
    for cell_index in (14, 15, 40, 43, 49, 70, 71):
        if cell_index >= len(notebook.get("cells", [])):
            continue
        cell = notebook["cells"][cell_index]
        for output in cell.get("outputs", []):
            image_data = output.get("data", {}).get("image/png")
            if image_data:
                encoded_image = image_data[0] if isinstance(image_data, list) else image_data
                figures[cell_index] = base64.b64decode(encoded_image)
                break
    return figures


def show_notebook_figure(figures, cell_index, caption):
    """Display a notebook's saved PNG output without regenerating its plot."""
    image = figures.get(cell_index)
    if image is None:
        st.info(f"The saved notebook image for this plot is unavailable (cell {cell_index}).")
        return False
    st.image(image, caption=caption, use_container_width=False)
    return True


def page_analytics(dataset):
    st.markdown('<div class="hero"><p>Notebook figures and evaluation results from the saved student dataset and trained models.</p></div>', unsafe_allow_html=True)
    evaluation = None
    dnn_scores = None
    rf_scores = None
    try:
        evaluation = load_evaluation_data()
        if evaluation is not None:
            _, actual, dnn_pred, rf_pred = evaluation
            dnn_scores = metric_bundle(actual, dnn_pred)
            if rf_pred is not None:
                rf_scores = metric_bundle(actual, rf_pred)
    except Exception as exc:
        st.warning(f"Saved models could not be evaluated: {exc}")

    st.caption("R-squared Score (%) = R-squared x 100. This is explained variance, not classification accuracy.")
    dnn_performance, rf_performance = st.columns(2, gap="medium")
    with dnn_performance:
        st.metric(
            "DNN Performance (%)",
            f"{dnn_scores['R-squared'] * 100:.2f}%" if dnn_scores is not None else "Unavailable",
        )
    with rf_performance:
        st.metric(
            "Random Forest Performance (%)",
            f"{rf_scores['R-squared'] * 100:.2f}%" if rf_scores is not None else "Unavailable",
        )

    st.subheader("Evaluation Metrics")
    if dnn_scores is not None:
        metric_rows = {"DNN": dnn_scores}
        if rf_scores is not None:
            metric_rows["Random Forest"] = rf_scores
        comparison_metrics = pd.DataFrame(metric_rows).T
        st.dataframe(comparison_metrics.style.format("{:.3f}"), use_container_width=True)
        st.caption("Both models are evaluated on the same saved test samples and labels.")
    else:
        st.info("Saved test data or model predictions are unavailable, so evaluation metrics cannot be calculated.")

    try:
        figures = load_notebook_figures()
    except Exception as exc:
        figures = {}
        st.warning(f"Saved notebook figures could not be loaded: {exc}")

    st.subheader("Dataset Analysis")
    show_notebook_figure(figures, 14, "Notebook figure: Distribution of Final Grades")
    show_notebook_figure(figures, 15, "Notebook figure: Feature Correlation Heatmap")

    st.subheader("DNN Training Analysis")
    loss_col, optimized_loss_col = st.columns(2, gap="medium")
    with loss_col:
        show_notebook_figure(figures, 40, "Notebook figure: DNN Training vs Validation Loss")
    with optimized_loss_col:
        show_notebook_figure(figures, 49, "Notebook figure: Optimized DNN Training vs Validation Loss")
    st.caption("The notebook logs training and validation MAE, but does not create or save an MAE curve.")

    st.subheader("Model Prediction Analysis")
    dnn_prediction_col, model_prediction_col = st.columns(2, gap="medium")
    with dnn_prediction_col:
        show_notebook_figure(figures, 43, "Notebook figure: Actual vs Predicted Student Grades")
    with model_prediction_col:
        show_notebook_figure(figures, 71, "Notebook figure: Actual vs Predicted Grades for DNN and Random Forest")

    st.subheader("Model Comparison")
    st.caption("The notebook comparison plot shows MAE and RMSE.")
    show_notebook_figure(figures, 70, "Notebook figure: DNN vs Random Forest (MAE and RMSE)")

def page_about(dataset):
    st.markdown('<div class="hero"><p>An academic project that estimates final Portuguese course grades from student profile and progress data.</p></div>', unsafe_allow_html=True)
    st.subheader("Project overview")
    st.write("This project uses supervised learning to estimate a student's final course grade (G3). It demonstrates data exploration, preprocessing, model development, evaluation, and deployment in one Streamlit application.")
    st.subheader("Problem statement")
    st.write("Historical student records can help identify patterns associated with academic outcomes. These estimates can support discussion, but they should not replace professional judgment or individual support.")
    st.subheader("Objectives")
    st.markdown("- Estimate final academic grades from student information.\n- Explore factors associated with performance.\n- Evaluate a Deep Neural Network and compare it with a Random Forest regressor.\n- Present the saved models through an accessible dashboard.")
    st.subheader("Dataset")
    if dataset is not None:
        st.write(f"The loaded Portuguese course dataset contains **{len(dataset):,} students** and **{dataset.shape[1]} columns**, including the G3 target. The application form collects **{len(FEATURES)} input features**.")
    else:
        st.write("The Portuguese course dataset was not available when this page loaded.")
    st.write("Inputs cover student and family background, school support, study and lifestyle factors, absences, and the first- and second-period grades (G1 and G2).")
    st.subheader("Workflow")
    st.write("Dataset exploration, preprocessing, train-test split, DNN and Random Forest training, evaluation on saved test data, and dashboard integration.")
    st.subheader("Models and methodology")
    st.write("The DNN is a feed-forward regression model with hidden layers of 64, 32, and 16 ReLU units and one linear output. The notebook specifies Adam with a 0.001 learning rate, mean squared error loss, mean absolute error as a metric, and early stopping on validation loss with patience 15. The dashboard loads the saved model without retraining it.")
    st.write("The Random Forest regressor combines decision trees trained on bootstrapped samples and feature subsets. Its predictions are evaluated on the same saved test split as the DNN.")
    st.subheader("Prediction pipeline")
    st.write("The form creates a row with the original 32 feature names and order. The saved preprocessing pipeline applies its fitted encoding and scaling, then the DNN estimates G3. The interface displays the estimate on the dataset's 0-20 grade scale and assigns a descriptive performance category.")
    st.subheader("Technology")
    st.write("Python, Streamlit, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Matplotlib, and Joblib.")
    st.subheader("Limitations")
    st.write("The dataset is modest and represents a specific course and population, so results may not generalize to other settings. Predictions are estimates, not causal explanations or guaranteed outcomes. Because G1 and G2 are inputs, the model requires those grades and is not an early-term predictor.")


def render_kpis(dataset, preprocessor, dnn, rf):
    """Display concise project metrics from the loaded dataset and model assets."""
    raw_feature_names = getattr(preprocessor, "feature_names_in_", FEATURES)
    student_count = f"{len(dataset):,}" if dataset is not None else "Unavailable"
    model_count = int(dnn is not None) + int(rf is not None)
    metrics = [
        ("Total Students", student_count),
        ("Input Features", str(len(raw_feature_names))),
        ("Models Compared", str(model_count)),
        ("Prediction Target", "G3 / 20"),
    ]
    columns = st.columns(4, gap="medium")
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def main():
    try:
        dnn, preprocessor, rf = load_assets()
    except Exception as exc:
        st.error(f"Could not load the saved project artifacts: {exc}")
        st.info("Confirm that models/student_performance_dnn.keras and models/preprocessor.pkl exist and that project dependencies are installed.")
        st.stop()
    dataset = load_dataset()
    with st.sidebar:
        st.markdown("## Student Performance")
        st.caption("PORTUGUESE COURSE")
        selection = st.radio("Navigate", NAV_ITEMS, label_visibility="collapsed")
        st.divider()
        st.caption("Grade scale: 0-20")
    page_titles = {
        NAV_ITEMS[0]: "Prediction",
        NAV_ITEMS[1]: "Model Analytics",
        NAV_ITEMS[2]: "About Project",
    }
    title = page_titles[selection]
    st.markdown(
        f'<div class="topline"><span class="topline-title">{title}</span></div>',
        unsafe_allow_html=True,
    )
    render_kpis(dataset, preprocessor, dnn, rf)
    if selection == NAV_ITEMS[0]:
        page_prediction(dnn, preprocessor)
    elif selection == NAV_ITEMS[1]:
        page_analytics(dataset)
    else:
        page_about(dataset)


if __name__ == "__main__":
    main()
