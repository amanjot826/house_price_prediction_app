# importing the required libraries
# streamlit, flask, node.js, Django, Dash, FastAPI are primarily backend or full-stack frameworks
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# loading the dataset
cal = fetch_california_housing()
df = pd.DataFrame(data = cal.data, columns = cal.feature_names)
df['price'] = cal.target
df.head()

# naming the app
st.title("California House Price Prediction for XYZ Brokerage Company")

# providing data overview
st.subheader("Data Overview")
st.dataframe(df.head(5))

# splitting the dataset into train and test
x = df.drop(['price'], axis = 1) 
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# standardizing the data
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.fit_transform(x_test)

# model selection
st.subheader("## Select a Model")
model = st.selectbox("Options", ["Linear Regression", "Ridge", "Lasso", "ElasticNet"])

# initializing the model
models = {"Linear Regression": LinearRegression(),
          "Ridge": Ridge(alpha = 0.01),
          "Lasso": Lasso(alpha = 0.001),
          "ElasticNet": ElasticNet(alpha = 0.001)}

# training the selected model and predicting the values
selected_model = models[model]
selected_model.fit(x_train_sc, y_train)
y_pred = selected_model.predict(x_test_sc)

# evaluating the model using metrics
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

# displaying the metrics of the selected model
st.write("Test MSE:", test_mse)
st.write("Test MAE:", test_mae)
st.write("Test RMSE:", test_rmse)
st.write("Test R2:", test_r2)

# prompt for the user to enter the input values
st.write("Enter the input values to predict the house price:")
user_input = {}
for feature in x.columns:
    user_input[feature] = st.number_input(feature)

data = pd.DataFrame(user_input, index=[0])
user_input_sc = scaler.transform(data)
predicted_data = selected_model.predict(user_input_sc)
st.write("Predicted house price is", predicted_data)









