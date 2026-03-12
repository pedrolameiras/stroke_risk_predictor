from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import NUM_COLS_TO_SCALE, load_raw_data, train_and_save

st.set_page_config(
    page_title="Análise de Incidência de AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "healthcare-dataset-stroke-data.csv"

GENDER_MAP = {"Male": 0, "Female": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}
RESIDENCE_MAP = {"Rural": 0, "Urban": 1}
SMOKING_MAP = {"Unknown": 0, "Never smoked": 1, "Formerly smoked": 2, "Smokes": 3}
WORK_MAP = {
    "Children": 0,
    "Private": 1,
    "Self-employed": 2,
    "Govt job": 3,
    "Other": 4,
    "Never worked": 5,
}


@st.cache_resource(show_spinner=True)
def get_artifacts():
    return train_and_save(force_retrain=False)


@st.cache_data(show_spinner=False)
def get_descriptive_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
        section[data-testid="stSidebar"] {background: #f5f7fb;}
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: 0 6px 18px rgba(0,0,0,.06);
            border: 1px solid #e9eef5;
            text-align: center;
            min-height: 110px;
        }
        .metric-number-blue {font-size: 2rem; font-weight: 800; color: #d65158; margin: 0;}
        .metric-number-orange {font-size: 2rem; font-weight: 800; color: #d89b48; margin: 0;}
        .metric-label {font-size: 1.1rem; color: #3a3a3a; margin-top: 4px;}
        .panel-title {font-size: 2.2rem; font-weight: 800; color: #6288bc; margin-bottom: .6rem;}
        .result-box {
            border: 1px solid #dbe5ef;
            border-radius: 16px;
            padding: 18px;
            background: #ffffff;
            min-height: 170px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        }
        .banner-wait {font-size: 2.3rem; font-weight: 900; color: #334155;}
        .banner-ok {font-size: 2.3rem; font-weight: 900; color: #1b5e20;}
        .banner-risk {font-size: 2.3rem; font-weight: 900; color: #d32f2f;}
        .hero-left {
            background: #2fa8ea;
            padding: 28px;
            border-radius: 20px;
            min-height: 100%;
        }
        .hero-right {
            background: #ffffff;
            padding: 28px;
            border-radius: 20px;
            min-height: 100%;
            border: 1px solid #dbe5ef;
        }
        .hero-left h1 {color: white; margin-bottom: 0.3rem;}
        .hero-left p {color: #eaf7ff;}
        .card-surface {
            background: #ffffff;
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #d7e7f4;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        }
        .small-note {font-size: .96rem; color: #4b5563;}
        .footer-note {margin-top: 1.5rem; padding-top: .75rem; border-top: 1px solid #e5e7eb; font-size: .95rem; color: #6b7280; text-align: center;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title("Menu")
    page = st.sidebar.radio(
        "Escolhe a área",
        ["Modelo preditivo", "Análise descritiva", "Métricas do treino"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "A aplicação treina automaticamente o modelo na primeira execução e guarda os artefactos na pasta artifacts/."
    )
    return page


def render_model_page(artifacts: dict):
    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown("<div class='hero-left'>", unsafe_allow_html=True)
        st.markdown("<h1>Stroke Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p>Fill the form and click Predict Stroke.</p>", unsafe_allow_html=True)
        st.markdown("<div class='card-surface'>", unsafe_allow_html=True)

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
                hypertension = st.selectbox("Hypertension", list(YES_NO_MAP.keys()))
                ever_married = st.selectbox("Ever Married", list(YES_NO_MAP.keys()), index=1)
                work_type = st.selectbox("Work Type", list(WORK_MAP.keys()), index=1)
                avg_glucose_level = st.number_input(
                    "Average Glucose Level (> 0)", min_value=0.01, value=120.50, step=0.01
                )
            with c2:
                age = st.number_input("Age (1 to 100)", min_value=1, max_value=100, value=45, step=1)
                heart_disease = st.selectbox("Heart Disease", list(YES_NO_MAP.keys()))
                residence_type = st.selectbox("Residence Type", list(RESIDENCE_MAP.keys()), index=1)
                smoking_status = st.selectbox("Smoking Status", list(SMOKING_MAP.keys()), index=1)
                bmi = st.number_input("BMI (> 0)", min_value=0.01, value=27.30, step=0.01)

            submitted = st.form_submit_button("Predict Stroke", use_container_width=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='hero-right'>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='margin-top:0;'>Stroke <span style='background:#ffe600;padding:0 8px;border-radius:6px;'>risk prediction</span></h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='small-note'>This app predicts stroke risk from patient clinical data using a trained machine learning model. It returns a risk label and the estimated probability. Use it for screening support, not as a final medical diagnosis.</p>",
            unsafe_allow_html=True,
        )

        if submitted:
            input_df = pd.DataFrame(
                [
                    {
                        "gender": GENDER_MAP[gender],
                        "age": int(age),
                        "hypertension": YES_NO_MAP[hypertension],
                        "heart_disease": YES_NO_MAP[heart_disease],
                        "ever_married": YES_NO_MAP[ever_married],
                        "work_type": WORK_MAP[work_type],
                        "Residence_type": RESIDENCE_MAP[residence_type],
                        "avg_glucose_level": float(avg_glucose_level),
                        "bmi": float(bmi),
                        "smoking_status": SMOKING_MAP[smoking_status],
                    }
                ]
            )
            input_df = input_df.reindex(columns=artifacts["feature_order"])
            input_scaled = input_df.copy()
            input_scaled[NUM_COLS_TO_SCALE] = artifacts["scaler"].transform(input_scaled[NUM_COLS_TO_SCALE])

            pred = int(artifacts["model"].predict(input_scaled)[0])
            proba = float(artifacts["model"].predict_proba(input_scaled)[0, 1])

            banner_class = "banner-risk" if pred == 1 else "banner-ok"
            banner_text = "STROKE RISK"
            st.markdown(f"<div class='{banner_class}'>{banner_text}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:1.2rem;'>Probability: {proba * 100:.2f}%</p>", unsafe_allow_html=True)
            st.progress(min(max(proba, 0.0), 1.0))
        else:
            st.markdown("<div class='banner-wait'>Waiting for prediction...</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:1.2rem;'>Probability: -</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def build_age_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 18, 40, 60, 80, 300]
    labels = ["0-17", "18-39", "40-59", "60-79", "80+"]
    temp = df.copy()
    temp["Faixa etária"] = pd.cut(temp["age"], bins=bins, labels=labels, right=False)
    table = pd.crosstab(temp["stroke"].map({1: "sim", 0: "não"}), temp["Faixa etária"])
    return table.reset_index().rename(columns={"stroke": "AVC"})


def render_descriptive_page():
    df = get_descriptive_data()
    st.markdown("<div class='panel-title'>Análise Descritiva de Incidência de AVC</div>", unsafe_allow_html=True)

    metric1, metric2, _ = st.columns([1, 1, 4])
    with metric1:
        st.markdown(
            f"<div class='metric-card'><p class='metric-number-blue'>{len(df)}</p><div class='metric-label'>participantes</div></div>",
            unsafe_allow_html=True,
        )
    with metric2:
        st.markdown(
            f"<div class='metric-card'><p class='metric-number-orange'>{int(df['stroke'].sum())}</p><div class='metric-label'>casos de AVC</div></div>",
            unsafe_allow_html=True,
        )

    age_table = build_age_table(df)
    gender_df = (
        df.groupby("gender", as_index=False)["stroke"]
        .mean()
        .replace({"gender": {"Male": "Male", "Female": "Female", "Other": "Other"}})
    )
    residence_df = (
        df.groupby(["stroke", "Residence_type"], as_index=False)
        .size()
        .replace({"stroke": {1: "sim", 0: "não"}})
        .rename(columns={"size": "Contagem", "stroke": "AVC"})
    )
    hypertension_df = df.groupby("hypertension", as_index=False)["stroke"].mean()
    hypertension_df["stroke"] = hypertension_df["stroke"] * 100
    hypertension_df["hypertension"] = hypertension_df["hypertension"].replace({0: "não", 1: "sim"})

    heart_df = df.groupby("heart_disease", as_index=False)["stroke"].mean()
    heart_df["stroke"] = heart_df["stroke"] * 100
    heart_df["heart_disease"] = heart_df["heart_disease"].replace({0: "não", 1: "sim"})

    smoking_order = ["formerly smoked", "smokes", "never smoked", "Unknown"]
    smoking_df = df.groupby("smoking_status", as_index=False)["stroke"].mean()
    smoking_df["% AVC"] = smoking_df["stroke"] * 100
    smoking_df = smoking_df.drop(columns=["stroke"])
    smoking_df["smoking_status"] = pd.Categorical(
        smoking_df["smoking_status"], categories=smoking_order, ordered=True
    )
    smoking_df = smoking_df.sort_values("smoking_status").dropna(subset=["smoking_status"])

    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ["Baixo peso", "Normal", "Sobrepeso", "Obesidade"]
    temp_bmi = df.copy()
    temp_bmi["Grupo IMC"] = pd.cut(temp_bmi["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)
    bmi_df = (
        temp_bmi.dropna(subset=["Grupo IMC"]).groupby("Grupo IMC", as_index=False).size().rename(columns={"size": "Contagem"})
    )

    row1_col1, row1_col2, row1_col3 = st.columns([1.35, 1.0, 1.15], gap="large")

    with row1_col1:
        st.markdown("##### Grupos de Idade")
        fig_age = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["AVC"] + age_table.columns[1:].tolist(), fill_color="#ffffff", align="center"),
                    cells=dict(values=[age_table[col] for col in age_table.columns], fill_color="#e0f3ef", align="center"),
                )
            ]
        )
        fig_age.update_layout(height=220, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig_age, use_container_width=True)

    with row1_col2:
        st.markdown("##### Género")
        fig_gender = px.pie(
            gender_df,
            values="stroke",
            names="gender",
            color="gender",
            color_discrete_map={"Female": "#e7c6df", "Male": "#96689f", "Other": "#cccccc"},
            hole=0.0,
        )
        fig_gender.update_traces(texttemplate="%{value:.2%}", hovertemplate="Género=%{label}<br>Média AVC=%{value:.4f}<extra></extra>")
        fig_gender.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Género")
        st.plotly_chart(fig_gender, use_container_width=True)

    with row1_col3:
        st.markdown("##### Tipo Residência")
        tabs = st.tabs(["AVC = sim", "AVC = não"])
        for avc_val, tab in zip(["sim", "não"], tabs):
            with tab:
                pie_df = residence_df[residence_df["AVC"] == avc_val]
                fig_res = px.pie(
                    pie_df,
                    values="Contagem",
                    names="Residence_type",
                    color="Residence_type",
                    color_discrete_map={"Rural": "#398f45", "Urban": "#a9d49b"},
                )
                fig_res.update_traces(textinfo="value", hovertemplate="%{label}: %{value}<extra></extra>")
                fig_res.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Tipo Residência")
                st.plotly_chart(fig_res, use_container_width=True)

    row2_col1, row2_col2, row2_col3 = st.columns([1.0, 1.0, 1.2], gap="large")

    with row2_col1:
        st.markdown("##### Hipertensão")
        fig_hyper = px.bar(
            hypertension_df,
            x="hypertension",
            y="stroke",
            text="stroke",
            color="hypertension",
            color_discrete_map={"não": "#df5858", "sim": "#df5858"},
        )
        fig_hyper.update_traces(texttemplate="%{y:.2f}%", hovertemplate="Hipertensão=%{x}<br>% AVC=%{y:.2f}%<extra></extra>")
        fig_hyper.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="% de AVC")
        st.plotly_chart(fig_hyper, use_container_width=True)

        st.markdown("##### Doenças Cardíacas")
        fig_heart = px.bar(
            heart_df,
            x="heart_disease",
            y="stroke",
            text="stroke",
            color="heart_disease",
            color_discrete_map={"não": "#e3c55e", "sim": "#e3c55e"},
        )
        fig_heart.update_traces(texttemplate="%{y:.2f}%", hovertemplate="Doença Cardíaca=%{x}<br>% AVC=%{y:.2f}%<extra></extra>")
        fig_heart.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="% de AVC")
        st.plotly_chart(fig_heart, use_container_width=True)

    with row2_col2:
        st.markdown("##### Antecedentes tabágicos")
        smoke_plot_df = smoking_df.copy()
        smoke_plot_df["stack"] = ""
        smoke_plot_df["label"] = smoke_plot_df["% AVC"].map(lambda v: f"{v:.2f}%".replace(".", ","))
        smoke_plot_df["smoking_status"] = smoke_plot_df["smoking_status"].astype(str)
        stack_order = ["Unknown", "never smoked", "smokes", "formerly smoked"]
        smoke_plot_df["smoking_status"] = pd.Categorical(
            smoke_plot_df["smoking_status"], categories=stack_order, ordered=True
        )
        smoke_plot_df = smoke_plot_df.sort_values("smoking_status")

        fig_smoke = px.bar(
            smoke_plot_df,
            x="stack",
            y="% AVC",
            color="smoking_status",
            text="label",
            category_orders={"smoking_status": stack_order},
            color_discrete_map={
                "formerly smoked": "#a94421",
                "smokes": "#f07d22",
                "never smoked": "#f3c17c",
                "Unknown": "#beb9b9",
            },
        )
        fig_smoke.update_layout(
            barmode="stack",
            height=460,
            xaxis_title="",
            yaxis_title="Média de AVC",
            showlegend=True,
            legend_title_text="Antecedentes tabágicos",
            legend_traceorder="reversed",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_smoke.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_smoke.update_yaxes(range=[0, 22], ticksuffix=",00%", dtick=2)
        fig_smoke.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12),
            hovertemplate="%{customdata[0]}<br>Média de AVC=%{y:.2f}%<extra></extra>",
            customdata=smoke_plot_df[["smoking_status"]],
        )
        st.plotly_chart(fig_smoke, use_container_width=True)

    with row2_col3:
        st.markdown("##### IMC")
        coords = pd.DataFrame(
            {
                "Grupo IMC": ["Obesidade", "Sobrepeso", "Normal", "Baixo peso"],
                "x": [0.28, 0.55, 0.84, 0.78],
                "y": [0.70, 0.24, 0.54, 0.34],
            }
        )
        bubble_df = bmi_df.merge(coords, on="Grupo IMC", how="left")
        fig_bmi = px.scatter(
            bubble_df,
            x="x",
            y="y",
            size="Contagem",
            color="Grupo IMC",
            color_discrete_map={
                "Baixo peso": "#c8dfed",
                "Normal": "#87b2d6",
                "Obesidade": "#356190",
                "Sobrepeso": "#4f7eab",
            },
            text="Grupo IMC",
            size_max=115,
        )
        fig_bmi.update_traces(
            textposition="middle center",
            hovertemplate="Grupo IMC=%{customdata[0]}<br>Contagem=%{marker.size}<extra></extra>",
            customdata=bubble_df[["Grupo IMC"]],
        )
        fig_bmi.update_layout(
            height=660,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            legend_title_text="Grupos IMC",
        )
        st.plotly_chart(fig_bmi, use_container_width=True)



def render_footer():
    st.markdown(
        "<div class='footer-note'>Discente: Eduarda Padrão, Mariana Paixão, Pedro Lameiras e Sara Urbano</div>",
        unsafe_allow_html=True,
    )

def render_metrics_page(artifacts: dict):
    st.markdown("<div class='panel-title'>Métricas do treino</div>", unsafe_allow_html=True)
    metrics = artifacts["metrics"]

    st.subheader("Accuracy média por validação cruzada")
    cv_df = pd.DataFrame(
        {"Modelo": list(metrics["cross_val_accuracy"].keys()), "Accuracy": list(metrics["cross_val_accuracy"].values())}
    )
    fig_cv = px.bar(cv_df, x="Modelo", y="Accuracy", text="Accuracy")
    fig_cv.update_traces(texttemplate="%{y:.4f}", hovertemplate="Modelo=%{x}<br>Accuracy=%{y:.4f}<extra></extra>")
    st.plotly_chart(fig_cv, use_container_width=True)

    st.subheader("Comparação final dos modelos")
    st.dataframe(metrics["evaluation_df"], use_container_width=True)

    st.subheader("Melhores hiperparâmetros do XGBoost otimizado")
    st.json(metrics["best_params"])

    st.subheader("Matriz de confusão do melhor modelo")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["No Stroke", "Stroke"], columns=["No Stroke", "Stroke"])
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    fig_cm.update_layout(xaxis_title="Predicted label", yaxis_title="True label")
    st.plotly_chart(fig_cm, use_container_width=True)


def main():
    inject_css()
    artifacts = get_artifacts()
    page = render_sidebar()

    if page == "Modelo preditivo":
        render_model_page(artifacts)
    elif page == "Análise descritiva":
        render_descriptive_page()
    else:
        render_metrics_page(artifacts)

    render_footer()


if __name__ == "__main__":
    main()
