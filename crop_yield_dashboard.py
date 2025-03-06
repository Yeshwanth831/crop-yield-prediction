import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.ensemble import RandomForestRegressor
import io

# Set page configuration
st.set_page_config(page_title="🌾 AI Crop Yield Dashboard", layout="wide")

# Load dataset
df = pd.read_csv('wheat_yield_data.csv')

# Train the model
X = df[['Rainfall_mm', 'Temperature_C', 'Soil_Nitrogen_mgkg']]
y = df['Yield_tons_ha']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Header with animation
st.title("🌾 Sustainable Agriculture AI Dashboard")
st.markdown("<style>h1 { text-align: center; color: green; }</style>", unsafe_allow_html=True)
st.success("Using AI for Better Farming & Sustainability! 🌍")

# Sidebar Inputs with animation
st.sidebar.header("Enter Environmental Conditions")
rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 200, 1000, 500)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 20.0, 30.0, 24.0, step=0.1)
soil_nitrogen = st.sidebar.slider("🧪 Soil Nitrogen (mg/kg)", 20, 40, 30)

# Animated progress bar
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

# Make Prediction
new_data = np.array([[rainfall, temperature, soil_nitrogen]])
prediction = model.predict(new_data)[0]

# Display Prediction with animation
st.subheader("🌱 Predicted Wheat Yield")
st.metric(label="Predicted Yield", value=f"{prediction:.2f} tons/ha")
st.balloons()

# Sustainability Recommendations with expanders
st.subheader("♻️ Sustainability Recommendations")
with st.expander("📌 Suggested Actions"):
    if prediction < 3.5:
        if rainfall < 450:
            st.write("- 🚰 **Increase Irrigation**: Use drip irrigation for 15% water savings.")
        if soil_nitrogen < 30:
            st.write("- 🌿 **Improve Soil Fertility**: Apply 25-35 kg/ha nitrogen fertilizer.")
    elif prediction >= 3.5 and rainfall > 700:
        st.write("- 💧 **Reduce Water Usage**: Optimize irrigation to save 10-15% water.")

# Feature Importance Visualization using Plotly
st.subheader("📊 Feature Importance")
feature_importance = model.feature_importances_
features = ['Rainfall', 'Temperature', 'Soil Nitrogen']
df_feature_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
fig = px.bar(df_feature_importance, x='Feature', y='Importance', title="What Factors Affect Yield the Most?", color='Importance', color_continuous_scale='greens')
st.plotly_chart(fig)

# Dataset Preview
st.subheader("📜 Reference Data")
st.dataframe(df.style.set_properties(**{'background-color': 'lightyellow', 'color': 'black'}))

# Downloadable Report with animation
st.subheader("📥 Download Your Report")
report = f"""
Crop Yield Prediction Report
----------------------------
Rainfall: {rainfall} mm
Temperature: {temperature} °C
Soil Nitrogen: {soil_nitrogen} mg/kg
Predicted Yield: {prediction:.2f} tons/ha

Recommendations:
"""
if prediction < 3.5:
    if rainfall < 450:
        report += "- Increase irrigation with drip systems.\n"
    if soil_nitrogen < 30:
        report += "- Apply 25-35 kg/ha nitrogen fertilizer.\n"
elif prediction >= 3.5 and rainfall > 700:
    report += "- Reduce irrigation to conserve water.\n"

buffer = io.StringIO()
buffer.write(report)
st.download_button("📩 Download Report", buffer.getvalue(), "yield_report.txt", "text/plain")

# Footer
st.markdown("<h4 style='text-align: center; color: darkgreen;'>🌾 Built with AI for Sustainable Farming 🌱</h4>", unsafe_allow_html=True)

