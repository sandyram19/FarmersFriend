import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from  crop_description import crop_descriptions
def add_sidebar():
    
    data=pd.read_csv('data/Crop_recommendation.csv')

    st.sidebar.header("N-P-K values")
      
    
    input_labels1 = {
    'N': st.sidebar.number_input("Nitrogen",min_value=float(data["N"].min()),max_value=float(data["N"].max()), value=float(data["N"].min()), key='N'),
    'P': st.sidebar.number_input("Phoshporous",min_value=float(data["P"].min()),max_value=float(data["P"].max()), value=float(data["P"].min()), key='P'),
    'K': st.sidebar.number_input("Potassium",min_value=float(data["K"].min()),max_value=float(data["K"].max()), value=float(data["K"].min()), key='K')}

    st.sidebar.header("Environmental Values")

    input_labels2={
    'temperature': st.sidebar.slider("Temperature",min_value=float(data["temperature"].min()),max_value=float(data["temperature"].max()),value=float(data["temperature"].mean()), key='temperature'),
    'humidity': st.sidebar.slider("Humidity",min_value=float(data["humidity"].min()),max_value=float(data["humidity"].max()), value=float(data["humidity"].mean()), key='humidity'),
    'ph': st.sidebar.slider("pH Label", min_value=0.0, max_value=14.0, value=data["ph"].mean(),key='ph'),
    'rainfall': st.sidebar.slider("Rainfall",min_value=float(data["humidity"].min()),max_value=float(data["rainfall"].max()),value=float(data["rainfall"].mean()), key='rainfall')
    }

    input_dict={}
    
    for i in input_labels1.keys():
        input_dict[i]=input_labels1[i]

    for i in input_labels2.keys():
        input_dict[i]=input_labels2[i]

    minimax={}
    for i in data.columns:
        minimax[i]=[data[i].min(),data[i].max()]
    
    return [input_dict,minimax]


def get_radar_chart(input_data):
    categories = list(input_data.keys())
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[input_data['N'],input_data['P'],input_data['K']],
      theta=categories,
      fill='toself',
      name='Soil Values'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[input_data['temperature'],input_data['humidity'],input_data['ph'],input_data['rainfall']],
      theta=categories,
      fill='toself',
      name='Environamental Values'
    ))
    

    fig.update_layout(
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
    showlegend=True
    )

    return fig
    
def add_predictions(input_data):
    model=pickle.load(open("model/model.pkl","rb"))
    scaler=pickle.load(open("model/scaler.pkl","rb"))
    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled=scaler.transform(input_array)

    prediction=model.predict(input_array_scaled)
    value=prediction[0]
    desc=crop_descriptions[value]
    
    st.subheader("Suitable Crop:")
    st.write("<button class='predicted'>", value, "</button>", unsafe_allow_html=True)
    st.write(f"Probability of {value} being a suitable crop is :",model.predict_proba(input_array_scaled).max())
    st.write(desc)

    return prediction[0]
def main():
    st.set_page_config(
        page_title="Crop Recommender",
        page_icon=":herb:",
        layout="wide",
        initial_sidebar_state="expanded"

    )

    with open("assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)

    input_data,mnx =add_sidebar()
    del mnx["label"]
    scaled_data=input_data.copy()
    for i in mnx.keys():
        mini,maxi=mnx[i]
        scaled_data[i]=(input_data[i]-mini)/(maxi-mini)

    # st.write(scaled_data)


    st.write("Crop Recommender")

    with st.container():
        st.title("Crop Recommender System")
        st.write("Find your Ideal crop for your field and maximise your Profits!")
    
    col1,col2=st.columns([4,1])

    with col1:
        radar_chart=get_radar_chart(scaled_data)
        st.plotly_chart(radar_chart)
    with col2:
        

        input_array=add_predictions(input_data)
        # st.write(input_array)


if __name__=='__main__':
    main()