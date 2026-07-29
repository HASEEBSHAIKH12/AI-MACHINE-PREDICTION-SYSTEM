import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
import tempfile
import os
from reportlab.lib.styles import getSampleStyleSheet 
# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SmartPredict AI",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/model.pkl")

# ----------------------------
# Prediction History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🤖 SmartPredict AI")
st.sidebar.success("✅ AI Model Loaded")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")

st.sidebar.write("""
### 👨‍💻 Developers

- Haseeb Shaikh
- Rafay Khan
- Mohsin Shah
""")

st.sidebar.markdown("---")

st.sidebar.info("""
Industrial Predictive Maintenance System

Machine Learning Project
""")

# ----------------------------
# Title
# ----------------------------
st.title("🏭 SmartPredict AI")
st.subheader("Industrial Predictive Maintenance System")

st.write("Enter Machine Details")

# ----------------------------
# Input Columns
# ----------------------------
left, right = st.columns(2)

with left:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=298.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        value=308.0
    )

with right:

    rpm = st.number_input(
        "Rotational Speed (RPM)",
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=10
    )

# ----------------------------
# Encode Machine Type
# ----------------------------
type_map = {
    "L":0,
    "M":1,
    "H":2
}

machine_type_encoded = type_map[machine_type]

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🚀 Predict Machine Status"):

    input_data = pd.DataFrame({

        "Type":[machine_type_encoded],

        "Air temperature [K]":[air_temp],

        "Process temperature [K]":[process_temp],

        "Rotational speed [rpm]":[rpm],

        "Torque [Nm]":[torque],

        "Tool wear [min]":[tool_wear]

    })
    # ----------------------------
    # Prediction
    # ----------------------------
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    healthy_prob = probability[0][0] * 100
    failure_prob = probability[0][1] * 100

    # Custom Threshold
    if failure_prob >= 30:
        prediction = [1]
    else:
        prediction = [0]

    health = max(0, min(100, int(100 - failure_prob)))

    st.markdown("---")

    # ----------------------------
    # Prediction Result
    # ----------------------------
    st.subheader("Prediction Result")

    if prediction[0] == 0:
        st.success("✅ Machine is Healthy")
    else:
        st.error("⚠ Machine Failure Predicted")

    # ----------------------------
    # Health Score
    # ----------------------------
    st.subheader("Machine Health Score")

    st.progress(health)
    st.write(f"**Health Score:** {health}%")

    if health >= 80:
     st.success("🟢 Excellent Machine Health")
    elif health >= 60:
     st.info("🟡 Good Machine Health")
    elif health >= 40:
     st.warning("🟠 Maintenance Recommended")
    else:
     st.error("🔴 Critical Machine Condition")

    # ----------------------------
    # Prediction Confidence
    # ----------------------------
    st.subheader("Prediction Confidence")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Healthy Probability", f"{healthy_prob:.2f}%")

    with c2:
        st.metric("Failure Probability", f"{failure_prob:.2f}%")

    # ----------------------------
    # Risk Level
    # ----------------------------
    st.subheader("Risk Level")

    if failure_prob < 20:
        st.success("🟢 LOW RISK")
    elif failure_prob < 50:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    # ----------------------------
    # AI Recommendation
    # ----------------------------
    st.subheader("AI Recommendation")

    if failure_prob < 20:
        st.success("Machine is operating normally.")
        st.write("✔ Continue production")
        st.write("✔ Routine maintenance only")
        st.write("✔ Monitor periodically")

    elif failure_prob < 50:
        st.warning("Machine requires maintenance.")
        st.write("✔ Inspect bearings")
        st.write("✔ Check lubrication")
        st.write("✔ Schedule maintenance soon")

    else:
        st.error("Critical Machine Condition")
        st.write("✔ Stop machine immediately")
        st.write("✔ Replace damaged components")
        st.write("✔ Contact maintenance team")
    st.session_state.history.append({

       "Machine Type": machine_type,

        "Health Score": health,

        "Failure %": round(failure_prob,2),

       "Risk":

       "Low" if failure_prob<20

       else "Medium" if failure_prob<50

        else "High"

    })
    # ----------------------------
    # Prediction Visualization
    # ----------------------------
    st.subheader("Prediction Visualization")

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        [healthy_prob, failure_prob],
        labels=["Healthy", "Failure"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.02, 0.08)
    )

    ax.axis("equal")

    st.pyplot(fig)

    # ----------------------------
    # Machine Input Summary
    # ----------------------------
    st.subheader("Machine Input Summary")

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

    st.dataframe(summary, use_container_width=True)

    # ----------------------------
    # Download Report
    # ----------------------------
    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "prediction_report.csv",
        "text/csv"
    )
    styles = getSampleStyleSheet()

    pdf_file= tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")     
    doc = SimpleDocTemplate(pdf_file.name)
    
    styles = getSampleStyleSheet()

    story=[]

    story.append(Paragraph("<b>SmartPredict AI Report</b>",styles["Title"]))

    story.append(Paragraph(f"Machine Type : {machine_type}",styles["BodyText"]))

    story.append(Paragraph(f"Health Score : {health}%",styles["BodyText"]))

    story.append(Paragraph(f"Failure Probability : {failure_prob:.2f}%",styles["BodyText"]))

    story.append(Paragraph(f"Risk Level : {'High' if failure_prob>50 else 'Medium' if failure_prob>20 else 'Low'}",styles["BodyText"]))

    doc.build(story)

    with open(pdf_file.name,"rb") as f:

     st.download_button(

        "📄 Download PDF Report",

        f,

        file_name="SmartPredict_Report.pdf"

    )

    os.unlink(pdf_file.name)
    st.subheader("Feature Importance")

    feature_data = pd.DataFrame({
      "Feature": [
          "Torque",
          "RPM",
          "Tool Wear",
          "Air Temp",
          "Process Temp",
          "Type"
        ],
        "Importance": [
          0.33,
          0.28,
          0.21,
          0.08,
          0.05,
          0.04
        ]
   })

    st.bar_chart(
      feature_data.set_index("Feature")
)
    # ----------------------------
    # Prediction History
    # ----------------------------

    st.markdown("---")
 
    st.subheader("Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st. dataframe(history_df, use_container_width=True)
    # ----------------------------
    # Model Information
    # ----------------------------
    st.markdown("---")

    st.subheader("Model Information")

    st.info(f"""
🤖 **Model:** Random Forest Classifier

📊 **Accuracy:** 94.85%

📁 **Dataset:** AI4I 2020 Predictive Maintenance

⚙️ **Custom Failure Threshold:** 30%

🧠 **Prediction Based On:** Machine Type, Temperature, RPM, Torque and Tool Wear
""")