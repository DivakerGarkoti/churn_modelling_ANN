# Customer Churn Prediction (ANN + Streamlit)

This project predicts whether a customer is likely to **churn** (leave a service) using an **Artificial Neural Network (ANN)**. It is built with **TensorFlow/Keras** for the model and **Streamlit** for a simple and interactive web interface.

---

## About

Customer churn is a key business metric that helps organizations retain valuable customers. This project uses a deep learning model trained on structured bank customer data to identify potential churn cases. The front end is built using Streamlit to allow users to input customer details and instantly get churn predictions.

---

## Features

- Predict customer churn using key indicators like age, credit score, balance, salary, etc.
- Backend powered by an ANN built with TensorFlow/Keras
- Clean and interactive frontend built using Streamlit
- Preprocessing includes Label Encoding, One-Hot Encoding, and Feature Scaling
- Model, encoders, and scaler are saved and loaded using Pickle
