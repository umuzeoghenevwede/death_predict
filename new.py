import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.header('Heart')

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#specify the feature
df = data[['age', 'anaemia', 'diabetes','high_blood_pressure', 'sex','smoking', 'time', 'platelets','DEATH_EVENT']].dropna()
x = df[['age', 'anaemia', 'diabetes','high_blood_pressure', 'sex','smoking', 'time','platelets']]

y = df['DEATH_EVENT']


#30% of the dataset is for testing and 70% of the dataset is for training
feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.3)

model =LogisticRegression()
model.fit(feature_train, target_train)



#Create a sidebar for my features
st.sidebar.header('Heart Failure')
st.sidebar.subheader('First Heart')

age = st.sidebar.number_input('Age of person',min_value=0)
anaemia = st.sidebar.number_input('Anaemia rate',min_value=0)
diabetes = st.sidebar.slider('Diabetes rate',0,1,0)
high_blood_pressure = st.sidebar.slider('high_blood_pressure rate',1,0,1)
sex = st.sidebar.slider('Sex rate',0,1,0)
smoking = st.sidebar.number_input('smoking rate',min_value=0)
time = st.sidebar.slider('time',4,285,2)
platelets = st.sidebar.number_input('platelets',min_value=0)


features = {'age': [age],
            'anaemia': [anaemia],
            'diabetes': [diabetes],
            'high_blood_pressure': [ high_blood_pressure],
            'sex': [sex],
             'smoking':[smoking],
            'time': [time],
            'platelets':[platelets]}



input_features = pd.DataFrame(features)
st.dataframe(features,width=1000)

#prediction code
if st.button('prediction on DEATH_EVENT'):
    prediction = model.predict(input_features)
    st.write('The prediction on the DEATH_EVENT for this patient is:',prediction)
