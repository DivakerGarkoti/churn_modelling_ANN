import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Loading Models
with open("le.pkl","rb")as file:
    le=pickle.load(file)
with open("scaler.pkl","rb")as file:
    scaler=pickle.load(file)
with open("ohe.pkl","rb")as file:
    ohe=pickle.load(file)
model=tf.keras.models.load_model("model.h5")

# streamlit app
st.title("Customer Churn Prediction")
geography=st.selectbox("Geography",ohe.categories_[0])
gender=st.selectbox("Gender",le.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member= st.selectbox('Is Active Member',[0,1])

#inputs
new_data=pd.DataFrame({'CreditScore':[credit_score],
          'Geography':[geography],
          'Gender':[le.transform([gender])[0]],
          'Age':[age],
          'Tenure':[tenure],
          'Balance':[balance],
          'NumOfProducts':[num_of_products],
          'HasCrCard':[has_cr_card],
          'IsActiveMember':[is_active_member],
          'EstimatedSalary':[estimated_salary]})

# data preprocessing
geo=ohe.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo,columns=ohe.get_feature_names_out())
new_data=new_data.drop(columns=["Geography"],axis=1)
new_data=pd.concat([new_data,geo_df])
new_data=scaler.transform(new_data)

#prediction
prediction=model.predict(new_data)

#checking conditions
if prediction>=0.5:
    st.write("Churn")
else:
    st.write("Not Churn")
