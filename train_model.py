from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "healthcare-dataset-stroke-data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "stroke_best_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "stroke_scaler.joblib"
FEATURE_ORDER_PATH = ARTIFACTS_DIR / "stroke_feature_order.joblib"
BMI_MEAN_PATH = ARTIFACTS_DIR / "stroke_bmi_mean.joblib"
METRICS_PATH = ARTIFACTS_DIR / "stroke_metrics.joblib"

NUM_COLS_TO_SCALE = ["age", "avg_glucose_level", "bmi"]


def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei a base de dados em: {DATA_PATH}. "
            "Coloca o ficheiro healthcare-dataset-stroke-data.csv na mesma pasta do projeto."
        )
    return pd.read_csv(DATA_PATH)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = df.copy()
    bmi_mean = float(df["bmi"].mean())
    df["bmi"] = df["bmi"].fillna(bmi_mean)
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"].copy()

    df["gender"] = df["gender"].replace({"Male": 0, "Female": 1})
    df["Residence_type"] = df["Residence_type"].replace({"Rural": 0, "Urban": 1})
    df["smoking_status"] = df["smoking_status"].replace(
        {"Unknown": 0, "never smoked": 1, "formerly smoked": 2, "smokes": 3}
    )
    df["ever_married"] = df["ever_married"].replace({"No": 0, "Yes": 1})
    df["work_type"] = df["work_type"].replace(
        {
            "children": 0,
            "Private": 1,
            "Self-employed": 2,
            "Govt_job": 3,
            "Other": 4,
            "Never_worked": 5,
        }
    )

    for col in ["age", "gender", "Residence_type", "smoking_status", "ever_married", "work_type"]:
        df[col] = df[col].astype(int)

    return df, bmi_mean


def train_and_save(force_retrain: bool = False) -> dict:
    if not force_retrain and all(
        path.exists()
        for path in [MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH, BMI_MEAN_PATH, METRICS_PATH]
    ):
        return load_artifacts()

    df = load_raw_data()
    prepared_df, bmi_mean = prepare_data(df)

    y = prepared_df["stroke"]
    X = prepared_df.drop(columns=["stroke"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }

    cross_val_accuracy = {}
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")
        cross_val_accuracy[nome] = float(scores.mean())

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUM_COLS_TO_SCALE] = scaler.fit_transform(X_scaled[NUM_COLS_TO_SCALE])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    for modelo in modelos.values():
        modelo.fit(X_train, y_train)

    peso_avc_max = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "scale_pos_weight": [1, 5, 10, 15, peso_avc_max],
    }

    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, tree_method="hist", verbosity=0)

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=15,
        cv=skf,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    models_to_evaluate = {
        "Decision Tree": modelos["Decision Tree"],
        "Random Forest": modelos["Random Forest"],
        "XGBoost (baseline)": modelos["XGBoost"],
        "XGBoost otimizado": best_model,
    }

    evaluation_rows = []
    for nome, modelo in models_to_evaluate.items():
        y_pred = modelo.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
        evaluation_rows.append(
            {
                "Modelo": nome,
                "Accuracy": float(report["accuracy"]),
                "Precision (AVC)": float(report["1"]["precision"]),
                "Recall (AVC)": float(report["1"]["recall"]),
                "F1-Score (AVC)": float(report["1"]["f1-score"]),
                "ROC-AUC": float(auc),
            }
        )

    y_pred_best = best_model.predict(X_test)
    metrics = {
        "cross_val_accuracy": cross_val_accuracy,
        "evaluation_df": pd.DataFrame(evaluation_rows),
        "best_params": random_search.best_params_,
        "confusion_matrix": confusion_matrix(y_test, y_pred_best),
    }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURE_ORDER_PATH)
    joblib.dump(bmi_mean, BMI_MEAN_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return {
        "model": best_model,
        "scaler": scaler,
        "feature_order": list(X.columns),
        "bmi_mean": bmi_mean,
        "metrics": metrics,
    }


def load_artifacts() -> dict:
    return {
        "model": joblib.load(MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "feature_order": joblib.load(FEATURE_ORDER_PATH),
        "bmi_mean": joblib.load(BMI_MEAN_PATH),
        "metrics": joblib.load(METRICS_PATH),
    }


if __name__ == "__main__":
    train_and_save(force_retrain=True)
    print("Treino concluído e artefactos gravados na pasta artifacts/.")
