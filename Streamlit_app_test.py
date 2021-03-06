# Inorder to access Frontend web app, Run "streamlit run Streamlit_app.py"
import streamlit as st
import requests
import json
import joblib
import numpy as np
import pandas as pd
import shap
import pickle
import streamlit.components.v1 as components
import plotly.express as px

with open('score_objects2.pkl', 'rb') as handle:
    clf_xgb_w, explainer_xgb = pickle.load(handle)  
X_test=pd.read_csv('X_train_ech.csv',index_col='Unnamed: 0')
z_1=pd.read_csv('z_1.csv',index_col='Unnamed: 0')
z_0=pd.read_csv('z_0.csv',index_col='Unnamed: 0')
st.set_page_config(layout="wide")

def run():
    st.header("Dashboard P7 - Bank Data Analysis !")
    test_id = st.sidebar.text_input("Id Client")
    column = st.selectbox("Colonne", ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
       'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'])
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

        #R??cup??ration de la pr??diction et du score
        response = requests.post("https://p7-final.herokuapp.com/predict", json=data)
        prediction=json.loads(response.text)
        st.success("Le score pr??dit par l'algorithme est : " +  str(prediction["score"]))
        st.success("La pr??diction de l'algo est : " + str(prediction["prediction"]))

        #Affichage du graphe Shap
        st.subheader('Model Prediction Interpretation Plot') 
        client=X_test.loc[int(test_id)] #recupere donnees client
        shap_values=explainer_xgb(client) #recuperer explainer 
        a=shap.force_plot(explainer_xgb.expected_value, shap_values.values[:], client) #recupere le force plot du client
        st_shap(a)  #affiche le force plot 
        
        st.subheader('Client :', str(test_id))

        #Pr??paration des donn??es - graphe : comparaison du client face aux Sets de clients ayant rembours?? ou non 
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
            #Affichage des donn??es du client 
            st.write("Donn??es du client ")
            client_2=client
            print('client 2 : ', client_2)
            st.dataframe(data=client_2) 
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()


