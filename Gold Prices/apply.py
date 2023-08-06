import numpy as np 
import pickle 
import streamlit as st
load_model= pickle.load(open('E:/save it/ml projects python/Gold Price/trained_model.sav','rb'))
def data(input_data):
    input_data_array=np.asarray(input_data)
    reshape_array=input_data_array.reshape(1,-1)

    prediction=load_model.predict(reshape_array)
    prediction
def main():
    st.title('GOLD Price predictor  web app')
    
    SPX=st.text_input("SPX")
    USO=st.text_input("USO")	
    SLV=st.text_input("SLV")
    EUR_USD=st.text_input("EUR_USD")
    pred= ''
    if st.button('GLD'):
        pred=data([SPX,USO, SLV, EUR_USD])
    st.success(pred)

if __name__=='__main__':
    main()    