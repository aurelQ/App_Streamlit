# Inorder to access Frontend web app, Run "streamlit run Streamlit_app.py"
import streamlit as st
import requests
import json
import joblib
import uvicorn
import numpy as np
import pandas as pd
import shap
import pickle
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

with open('../model/score_objects2.pkl', 'rb') as handle:
    clf_xgb_w, explainer_xgb = pickle.load(handle)  
X_test=pd.read_csv('../model/X_train_ech.csv',index_col='Unnamed: 0')
z_1=pd.read_csv('../model/z_1.csv',index_col='Unnamed: 0')
z_0=pd.read_csv('../model/z_0.csv',index_col='Unnamed: 0')
st.set_page_config(layout="wide")

def run():
    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.header("Dashboard P7 - Bank Data Analysis !")
    test_id = st.sidebar.text_input("Id Client")
    column = st.selectbox("Colonne", ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_INCOME_TOTAL'])
#    df_csv=pd.read_csv('../model/X_train_ech.csv')
#    df_csv=df_csv.iloc[test_id]

    data={
        'test_id': test_id
    }

    data_2={
    'column': column
    }
    
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
      
   
    if st.sidebar.button("Predict"):

        #Récupération de la prédiction et du score
        response = requests.post("https://p7-final.herokuapp.com/docs#/default/predict_predict_post", json=data)
        prediction=json.loads(response.text)
        st.success("Le score prédit par l'algorithme est : " +  str(prediction["score"]))
        st.success("La prédiction de l'algo est : " + str(prediction["prediction"]))

        #Affichage du graphe Shap
        st.subheader('Model Prediction Interpretation Plot') 
        client=X_test.loc[int(test_id)] #recupere donnees client
        shap_values=explainer_xgb(client) #recuperer explainer 
        a=shap.force_plot(explainer_xgb.expected_value, shap_values.values[:], client) #recupere le force plot du client
        st_shap(a)  #affiche le force plot 
        
        st.subheader('Client :', str(test_id))

        #Préparation des données - graphe : comparaison du client face aux Sets de clients ayant remboursé ou non 
        data_bar=[['Moyenne Z0',z_0[column].mean()],['Moyenne Z1',z_1[column].mean()],['Valeur client',client[column]]]  
        bar = pd.DataFrame(data_bar, columns=['Groupes de clients', str(column)])  
        fig = px.bar(        
        bar,
        x = 'Groupes de clients',
        y = str(column)
        )

        col1, col2= st.columns(2)
        with col1:
            st.write("Graphe client ")
            st.plotly_chart(fig) #Affichage du graphe
        with col2:
            #Affichage des données du client 
            st.write("Données du client ")
            client_2=client
            print('client 2 : ', client_2)
            st.dataframe(data=client_2) 
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()


