import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class GeminiClient:
    model_name: str = "gemini-1.5-flash"

    def _get_model(self):
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return None, "GOOGLE_API_KEY is not set."

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            return model, None
        except Exception as exc:  # noqa: BLE001
            return None, f"Gemini initialization failed: {exc}"

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        model, err = self._get_model()
        if err:
            return None, err

        try:
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            if text and text.strip():
                return text, None
            return None, "Gemini returned an empty response."
        except Exception as exc:  # noqa: BLE001
            return None, f"Gemini request failed: {exc}"


st.set_page_config(page_title="Cyber Security Threat Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Security Threat Agent")
st.caption("Complete local app with dashboard, model training, and Gemini-based analysis")


def read_dataset(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read CSV: {exc}")
        return None


def summarize_dataframe(df: pd.DataFrame, top_n: int = 5) -> str:
    lines = []
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append(f"Column Names: {', '.join(df.columns.tolist())}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        lines.append("\nNumeric Summary:")
        desc = df[numeric_cols].describe().round(3).to_string()
        lines.append(desc)

    if categorical_cols:
        lines.append("\nCategorical Top Values:")
        for col in categorical_cols[:8]:
            top_vals = df[col].astype(str).value_counts(dropna=False).head(top_n)
            lines.append(f"- {col}: {top_vals.to_dict()}")

    lines.append("\nSample Records:")
    lines.append(df.head(5).to_string(index=False))
    return "\n".join(lines)


def train_classifier(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        return None, "Selected target column does not exist."

    if df[target_col].nunique(dropna=True) < 2:
        return None, "Target must have at least 2 classes."

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)
    return {"model": model, "accuracy": acc, "report": report}, None


with st.sidebar:
    st.header("Controls")
    section = st.radio(
        "Go to",
        ["Dashboard", "Threat Analysis", "Model Training", "Incident Summary"],
    )
    st.markdown("---")
    uploaded = st.file_uploader("Upload threat dataset (CSV)", type=["csv"])


df = read_dataset(uploaded)
if df is None:
    st.info("Upload a CSV to unlock all features.")

if section == "Dashboard":
    st.subheader("📊 Dashboard")
    if df is None:
        st.warning("No dataset loaded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Cells", int(df.isna().sum().sum()))

        st.dataframe(df.head(25), use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if num_cols:
            chosen_num = st.selectbox("Numeric trend", num_cols)
            fig_hist = px.histogram(df, x=chosen_num, title=f"Distribution: {chosen_num}")
            st.plotly_chart(fig_hist, use_container_width=True)

        if cat_cols:
            chosen_cat = st.selectbox("Top categories", cat_cols)
            top_cat = (
                df[chosen_cat].astype(str).value_counts().head(15).reset_index()
            )
            top_cat.columns = [chosen_cat, "count"]
            fig_bar = px.bar(top_cat, x=chosen_cat, y="count", title=f"Top values: {chosen_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)

elif section == "Threat Analysis":
    st.subheader("🔍 Gemini Dataset Analysis")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        user_prompt = st.text_area(
            "What do you want to analyze?",
            value="Find the top threats, likely root causes, and 5 prioritized mitigations.",
            height=120,
        )

        if st.button("Analyze with Gemini", type="primary"):
            context = summarize_dataframe(df)
            full_prompt = (
                "You are a senior SOC analyst. Analyze this cybersecurity dataset summary and answer the user request. "
                "Return sections: Executive Summary, Key Findings, Attack Patterns, Risk Assessment, and Recommended Actions.\n\n"
                f"User Request:\n{user_prompt}\n\n"
                f"Dataset Context:\n{context}"
            )
            with st.spinner("Calling Gemini for analysis..."):
                text, err = GeminiClient().generate(full_prompt)
            if err:
                st.error(err)
                st.info("Set GOOGLE_API_KEY in your environment, then rerun the app.")
            else:
                st.success("Analysis generated")
                st.write(text)

elif section == "Model Training":
    st.subheader("🧠 Model Training")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        target_options = [c for c in df.columns if df[c].nunique(dropna=True) <= 100 and df[c].nunique(dropna=True) > 1]
        if not target_options:
            st.error("No suitable target column found. Add a categorical target column (e.g., attack_type).")
        else:
            target_col = st.selectbox("Select target column", target_options)
            if st.button("Start Training", type="primary"):
                with st.spinner("Training model..."):
                    result, err = train_classifier(df, target_col)
                if err:
                    st.error(err)
                else:
                    st.success(f"Training complete. Accuracy: {result['accuracy']:.4f}")
                    st.text("Classification report")
                    st.code(result["report"])

elif section == "Incident Summary":
    st.subheader("📋 Incident Summary")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        event_col = st.selectbox("Event ID column", options=df.columns)
        event_ids = df[event_col].astype(str).dropna().unique().tolist()
        selected_event = st.selectbox("Select event", options=event_ids[:2000])
        analyst_prompt = st.text_area(
            "Analyst prompt",
            value="Generate an incident summary, severity rationale, MITRE mapping, and remediation plan.",
            height=120,
        )

        if st.button("Generate Incident Summary", type="primary"):
            row = df[df[event_col].astype(str) == str(selected_event)].head(1)
            if row.empty:
                st.error("Selected event was not found.")
            else:
                incident_json = row.to_dict(orient="records")[0]
                prompt = (
                    "You are an incident response lead. Produce a structured report with sections: "
                    "Incident Overview, Indicators, Impact, MITRE ATT&CK Mapping, Containment, Eradication, Recovery, and Follow-up Tasks.\n\n"
                    f"Analyst request:\n{analyst_prompt}\n\n"
                    f"Incident data:\n{incident_json}"
                )
                with st.spinner("Generating summary with Gemini..."):
                    text, err = GeminiClient().generate(prompt)
                if err:
                    st.error(err)
                    st.json(incident_json)
                else:
                    st.write(text)

st.markdown("---")
st.caption("Tip: run with `streamlit run app.py` and set GOOGLE_API_KEY for Gemini features.")
