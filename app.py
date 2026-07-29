import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartPredict AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL AND ENCODER
# ============================================================

model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")


# ============================================================
# PREDICTION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 SmartPredict AI")

st.sidebar.success("✅ AI Model Loaded")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Project Team")

st.sidebar.write("""
- **Haseeb Shaikh**
- **Rafay Khan**
- **Mohsin Shah**
""")

st.sidebar.markdown("---")

st.sidebar.info("""
**Industrial Predictive Maintenance System**

Machine Learning Project
""")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🏭 SmartPredict AI")

st.subheader("Industrial Predictive Maintenance System")

st.write(
    "Enter machine details and click **Predict Machine Status**."
)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2)


with left:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )


    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=298.0
    )


    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=308.0
    )


with right:

    rpm = st.number_input(
        "Rotational Speed (RPM)",
        min_value=0,
        max_value=5000,
        value=1500
    )


    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )


    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=10
    )


# ============================================================
# ENCODE MACHINE TYPE
# ============================================================

machine_type_encoded = encoder.transform(
    [machine_type]
)[0]


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🚀 Predict Machine Status"):

    input_data = pd.DataFrame({
        
    

        "Type": [machine_type_encoded],

        "Air temperature [K]": [air_temp],

        "Process temperature [K]": [process_temp],

        "Rotational speed [rpm]": [rpm],

        "Torque [Nm]": [torque],

        "Tool wear [min]": [tool_wear]
    })

        # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)


    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    healthy_prob = probability[0][0] * 100

    failure_prob = probability[0][1] * 100


    # ========================================================
    # CUSTOM FAILURE THRESHOLD
    # ========================================================

    if failure_prob >= 30:

        prediction = [1]

    else:

        prediction = [0]


    # ========================================================
    # MACHINE HEALTH SCORE
    # ========================================================

    health = max(
        0,
        min(
            100,
            int(100 - failure_prob)
        )
    )


    st.markdown("---")


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader("🔍 Prediction Result")


    if prediction[0] == 0:

        st.success(
            "✅ Machine is Healthy"
        )

    else:

        st.error(
            "⚠️ Machine Failure Predicted"
        )


    # ========================================================
    # MACHINE HEALTH SCORE
    # ========================================================

    st.subheader("❤️ Machine Health Score")


    st.progress(health)


    st.write(
        f"**Health Score: {health}%**"
    )


    if health >= 80:

        st.success(
            "🟢 Excellent Machine Health"
        )

    elif health >= 60:

        st.info(
            "🟡 Good Machine Health"
        )

    elif health >= 40:

        st.warning(
            "🟠 Maintenance Recommended"
        )

    else:

        st.error(
            "🔴 Critical Machine Condition"
        )


    # ========================================================
    # PREDICTION CONFIDENCE
    # ========================================================

    st.subheader("📊 Prediction Confidence")


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Healthy Probability",
            f"{healthy_prob:.2f}%"
        )


    with c2:

        st.metric(
            "Failure Probability",
            f"{failure_prob:.2f}%"
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    st.subheader("⚠️ Risk Level")


    if failure_prob < 20:

        risk_level = "Low"

        st.success(
            "🟢 LOW RISK"
        )


    elif failure_prob < 50:

        risk_level = "Medium"

        st.warning(
            "🟡 MEDIUM RISK"
        )


    else:

        risk_level = "High"

        st.error(
            "🔴 HIGH RISK"
        )


    # ========================================================
    # AI MAINTENANCE RECOMMENDATION
    # ========================================================

    st.subheader("🧠 AI Maintenance Recommendation")


    if failure_prob < 20:

        st.success(
            "Machine is operating normally."
        )

        st.write(
            "✔ Continue production"
        )

        st.write(
            "✔ Perform routine maintenance"
        )

        st.write(
            "✔ Continue regular monitoring"
        )


    elif failure_prob < 50:

        st.warning(
            "Machine requires inspection and maintenance."
        )

        st.write(
            "✔ Inspect bearings"
        )

        st.write(
            "✔ Check lubrication"
        )

        st.write(
            "✔ Schedule maintenance soon"
        )


    else:

        st.error(
            "Critical machine condition detected."
        )

        st.write(
            "✔ Stop machine if necessary"
        )

        st.write(
            "✔ Inspect motor and bearings"
        )

        st.write(
            "✔ Check for excessive tool wear"
        )

        st.write(
            "✔ Contact maintenance team"
        )
        # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    st.session_state.history.append({

        "Machine Type": machine_type,

        "Health Score": health,

        "Healthy Probability": round(
            healthy_prob,
            2
        ),

        "Failure Probability": round(
            failure_prob,
            2
        ),

        "Risk Level": risk_level

    })


    # ========================================================
    # PREDICTION VISUALIZATION
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Prediction Visualization")


    fig, ax = plt.subplots(
        figsize=(5, 5)
    )


    ax.pie(

        [
            healthy_prob,
            failure_prob
        ],

        labels=[
            "Healthy",
            "Failure"
        ],

        autopct="%1.1f%%",

        startangle=90,

        explode=(
            0.02,
            0.08
        )

    )


    ax.axis("equal")


    st.pyplot(fig)


    # ========================================================
    # MACHINE INPUT SUMMARY
    # ========================================================

    st.subheader("📋 Machine Input Summary")


    summary = pd.DataFrame({

        "Feature": [

            "Machine Type",

            "Air Temperature (K)",

            "Process Temperature (K)",

            "Rotational Speed (RPM)",

            "Torque (Nm)",

            "Tool Wear (min)"

        ],

        "Value": [

            machine_type,

            air_temp,

            process_temp,

            rpm,

            torque,

            tool_wear

        ]

    })


    st.dataframe(

        summary,

        use_container_width=True

    )


    # ========================================================
    # CSV DOWNLOAD REPORT
    # ========================================================

    csv = summary.to_csv(

        index=False

    ).encode("utf-8")


    st.download_button(

        label="📥 Download CSV Report",

        data=csv,

        file_name="prediction_report.csv",

        mime="text/csv"

    )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.subheader("📄 PDF Report")


    styles = getSampleStyleSheet()


    pdf_file = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".pdf"

    )


    doc = SimpleDocTemplate(

        pdf_file.name

    )


    story = []


    story.append(

        Paragraph(

            "<b>SmartPredict AI Report</b>",

            styles["Title"]

        )

    )


    story.append(

        Paragraph(

            f"Machine Type: {machine_type}",

            styles["BodyText"]

        )

    )


    story.append(

        Paragraph(

            f"Health Score: {health}%",

            styles["BodyText"]

        )

    )


    story.append(

        Paragraph(

            f"Healthy Probability: {healthy_prob:.2f}%",

            styles["BodyText"]

        )

    )


    story.append(

        Paragraph(

            f"Failure Probability: {failure_prob:.2f}%",

            styles["BodyText"]

        )

    )


    story.append(

        Paragraph(

            f"Risk Level: {risk_level}",

            styles["BodyText"]

        )

    )


    doc.build(story)


    with open(

        pdf_file.name,

        "rb"

    ) as f:

        pdf_data = f.read()


    st.download_button(

        label="📄 Download PDF Report",

        data=pdf_data,

        file_name="SmartPredict_Report.pdf",

        mime="application/pdf"

    )


    os.unlink()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown("---")

    st.subheader("🧠 Feature Importance")


    feature_data = pd.DataFrame({

        "Feature": [

            "Type",

            "Air Temperature",

            "Process Temperature",

            "RPM",

            "Torque",

            "Tool Wear"

        ],

        "Importance": model.feature_importances_

    })


    st.bar_chart(

        feature_data.set_index(

            "Feature"

        )

    )


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    st.markdown("---")

    st.subheader("📜 Prediction History")


    history_df = pd.DataFrame(

        st.session_state.history

    )


    st.dataframe(

        history_df,

        use_container_width=True

    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader("ℹ️ Model Information")


    st.info("""

🤖 **Model:** Random Forest Classifier

📊 **Accuracy:** 94.85%

📁 **Dataset:** AI4I 2020 Predictive Maintenance

⚙️ **Custom Failure Threshold:** 30%

🧠 **Prediction Features:**

Machine Type, Air Temperature, Process Temperature,
Rotational Speed, Torque and Tool Wear.

""")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(

    "🏭 SmartPredict AI | Industrial Predictive Maintenance System"

)

st.caption(

    "Developed by Haseeb Shaikh, Rafay Khan and Mohsin Shah"

)
    