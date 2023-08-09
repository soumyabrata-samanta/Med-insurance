import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv("insurance.csv")

# Data preprocessing
data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
data.replace({'smoker': {'no': 1, 'yes': 0}}, inplace=True)
data.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = data.drop(columns="charges", axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

# Streamlit app
st.title("Insurance Cost Prediction")
st.sidebar.header("User Input")

# Sidebar inputs
age = st.sidebar.number_input("Age", value=30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", value=22.7)
children = st.sidebar.number_input("Number of Children", value=0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# Convert categorical input to numerical values
sex = 1 if sex == "Female" else 0
smoker = 0 if smoker == "Yes" else 1
region_mapping = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
region = region_mapping[region]

# Prepare input data
input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

# Make prediction
prediction = reg.predict(input_data)

# Confidence interval
confidence_interval = 0.95
prediction_interval = reg.predict(X_test)
prediction_std = np.std(prediction_interval)
confidence_factor = stats.t.ppf((1 + confidence_interval) / 2, len(X_test) - 1)
lower_bound = prediction - confidence_factor * prediction_std
upper_bound = prediction + confidence_factor * prediction_std
# Display prediction
st.subheader("Insurance Cost Prediction")
st.write("Predicted Insurance Cost:", round(prediction[0], 2))
st.write(f"Confidence Interval ({confidence_interval * 100:.1f}%):", (round(lower_bound[0], 2), round(upper_bound[0], 2)))

# Model evaluation
st.subheader("Model Evaluation")
training_data_prediction = reg.predict(X_train)
r2_train = metrics.r2_score(y_train, training_data_prediction)
st.write("R-squared (Training):", round(r2_train, 2))

test_data_prediction = reg.predict(X_test)
r2_test = metrics.r2_score(y_test, test_data_prediction)
st.write("R-squared (Test):", round(r2_test, 2))

# Visualization
st.subheader("Data Visualization")

# Age Distribution
fig_age, ax_age = plt.subplots(figsize=(8, 2))
sns.distplot(data['age'], ax=ax_age)
ax_age.set_title("Age Distribution")
st.pyplot(fig_age)

# Sex Distribution
fig_sex, ax_sex = plt.subplots(figsize=(8, 2))
sns.countplot(x="sex", data=data, ax=ax_sex)
ax_sex.set_title("Sex Distribution")
st.pyplot(fig_sex)

# BMI Distribution
fig_bmi, ax_bmi = plt.subplots(figsize=(8, 2))
sns.distplot(data['bmi'], ax=ax_bmi)
ax_bmi.set_title("BMI Distribution")
st.pyplot(fig_bmi)
