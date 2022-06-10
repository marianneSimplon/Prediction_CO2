import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

### Load / Preprocess / Predict test.csv dataset ###
MODEL_VERSION = 'pipeline.pkl'

# Load model
MODEL_PATH = os.path.join(os.getcwd(), 'modele',
                          MODEL_VERSION)  # path vers le modèle
MODEL = joblib.load(MODEL_PATH)  # chargement du modèle

###### STREAMLIT ######

### Header ###
st.title('Prédiction de l\'émission de CO2')

# Preprocessing des inputs


def preprocessNaN(input):
    # df = pd.read_csv(
    #     r'.\data\model.csv', delimiter=',', decimal='.')

    df = pd.read_csv(
        os.path.join(os.getcwd(), 'model.csv'), delimiter=',', decimal='.')

    for col in input:
        # get dtype for column
        dt = input[col].dtype
        # check if it is an object
        if dt == object:
            input[col].fillna("Rien", inplace=True)
        else:
            input[col].fillna(df[col].mean(), inplace=True)

    return input


def preprocess(input):
    input = input[['BuildingType',
                   'PrimaryPropertyType',
                   'NumberofFloors',
                   'PropertyGFAParking',
                   'LargestPropertyUseType',
                   'SecondLargestPropertyUseType',
                   'SecondLargestPropertyUseTypeGFA',
                   'SourceEUI(kBtu/sf)',
                   'SteamUse(kBtu)',
                   'Electricity(kBtu)']]

    return input


def predictCO2(input):
    pred = MODEL.predict(pd.DataFrame(input))
    st.write('Prédiction de TotalGHGEmissions :')
    if pred is np.ndarray:
        st.dataframe(pd.DataFrame(pred))
    else:
        st.write(pred)


### Upload test dataset ###
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    datas = pd.read_csv(uploaded_file)
    datas = preprocess(datas)

    # Si NaN
    if datas.isnull().values.any():
        datas = preprocessNaN(datas)

        ### Display button for prediction ###
    if st.button('Estimer TotalGHGEmissions'):
        predictCO2(datas)
