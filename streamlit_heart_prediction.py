import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model and scaler
heart_model = joblib.load("top_features_heart_model.pkl")
heart_scaler = joblib.load("top10_scaler.pkl")
# Define the features in order
risk_features = ["Smoking", "High Blood Pressure", "Family Heart Disease", "Age", 
                 "Triglyceride Level", "BMI", "Homocysteine Level", 
                 "Sleep Hours", "Cholesterol Level", "Stress Level_Medium"]

# Feature Input UI
st.set_page_config(page_title="Heart Disease Risk Prediction System", layout="centered")
st.title("❤️ Heart Disease Risk Prediction System")
st.subheader("Project Done by Samuel Nnamani a.k.a Sammyst The Analyst")
st.write("Enter the details below to predict your heart disease risk.")

user_input = {
    "Smoking": st.selectbox("Do you smoke?", ["No", "Yes"]) == "Yes",
    "High Blood Pressure": st.selectbox("Do you have high blood pressure?", ["No", "Yes"]) == "Yes",
    "Family Heart Disease": st.selectbox("Any family history of heart disease?", ["No", "Yes"]) == "Yes",
    "Age": st.slider("Age", 10, 100, 45),
    "Triglyceride Level": st.slider("Triglyceride Level (mg/dl)", 50, 500, 150),
    "BMI": st.slider("Body Mass Index (BMI)", 10.0, 50.0, 22.0),
    "Homocysteine Level": st.slider("Homocysteine Level (umol/L)", 5.0, 50.0, 15.0),
    "Sleep Hours": st.slider("Average Sleep per Night (hours)", 3, 12, 7),
    "Cholesterol Level": st.slider("Cholesterol Level (mg/dL)", 100, 300, 180),
    "Stress Level": st.selectbox("Stress Level", ["Low", "Medium", "High"])
}

if st.button("Predict Heart Risk"):
    # Convert input into Dataframe
    input_df = pd.DataFrame([user_input])
    input_df["Stress Level_Medium"] = 1 if input_df["Stress Level"].iloc[0] == "Medium" else 0
    input_df.drop(columns="Stress Level", inplace=True)

    # Reorder columns
    X_input = input_df[risk_features]

    # Normalize for scoring
    X_scaled = heart_scaler.transform(X_input)
    X_scaled_df = pd.DataFrame(X_scaled, columns=risk_features)

    # Define weights (from feature importance or predefined)
    weights_dict = {
        "Smoking": 0.011312,
        "High Blood Pressure": 0.010998,
        "Family Heart Disease": 0.015065,
        "Age": 0.071142,
        "Triglyceride Level": 0.084534,
        "BMI": 0.089601,
        "Homocysteine Level": 0.087650,
        "Sleep Hours": 0.091747,
        "Cholesterol Level": 0.081212,
        "Stress Level_Medium": 0.011241
    }

    # Compute risk score
    risk_score = sum(X_scaled_df[col].values[0] * weights_dict.get(col, 0) for col in X_scaled_df.columns)
    risk_score = round(risk_score * 100, 2)

    # Predict using the model
    prediction = heart_model.predict(X_scaled_df)
    # prediction = heart_model.predict(X_input)[0]
    prob = heart_model.predict_proba(X_input)[0][1]

    # Display the Results
    st.subheader("🔍 Prediction Result")
    st.success("✅ No Heart Disease Risk Detected" if prediction == 0 else "⚠️ Heart Disease Risk Detected")
    st.metric(label="Prediction Probability", value=f"{risk_score:.2f}")

    # Visualize Feature contributions
    st.subheader("📊 Feature Contributions to Heart Disease Risk Score")
    contributions = {
        col: X_scaled_df[col].values[0] * weights_dict.get(col, 0)
        for col in X_scaled_df.columns
    }

    fig = go.Figure(go.Bar(
        x=list(contributions.values()),
        y=list(contributions.keys()),
        orientation='h',
        marker=dict(color='crimson')
    ))
    fig.update_layout(
        title="Heart Disease Risk Feature Contributions",
        xaxis_title="Contribution (scaled & weighted)",
        yaxis_title="Features",
        height=500
    )
    st.plotly_chart(fig)

    # --------------------- HEALTH SCORING SYSTEM ------------------------
    st.subheader("💡 Patient Health Score")

    # Define weights for health score
    health_weights = {
        "Smoking": -1,
        "High Blood Pressure": -1,
        "Family Heart Disease": -1,
        "Age": -7,
        "Triglyceride Level": -8,
        "BMI": -9,
        "Homocysteine Level": -9,
        "Sleep Hours": -9,
        "Cholesterol Level": -8,
        "Stress Level_Medium": -1
    }

    # Normalize features again for the Health Scoring
    health_score_raw = 50    # Base score
    for feature in risk_features:
        val = X_scaled_df[feature].values[0]
        weight = health_weights.get(feature, 0)
        health_score_raw += val * weight

    health_score = np.clip(health_score_raw, 0, 100)

    # Health Score interpretation
    if health_score >= 80:
        health_msg = "🌟 Excellent"
    elif health_score >= 60:
        health_msg = "✅ Good"
    elif health_score >= 40:
        health_msg = "⚠️ Fair"
    else:
        health_msg = "❌ Poor"

    # Show score & Interpretation
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Health Score (0-100)", f"{health_score:.1f}", help="Reflects overall Lifestyle & Heart Health")
        st.write(f"**Status:** {health_msg}")

    # Plotly Gauge Chart
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "red"},
                    {"range": [40, 60], "color": "orange"},
                    {"range": [60, 80], "color": "lightgreen"},
                    {"range": [80, 100], "color": "green"},
                ]
            }
        ))
        st.plotly_chart(fig)