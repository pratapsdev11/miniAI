import numpy as np
import pickle 
import streamlit as st

kees=list(range(0,22))
targets={0: 'apple',
 1: 'banana',
 2: 'blackgram',
 3: 'chickpea',
 4: 'coconut',
 5: 'coffee',
 6: 'cotton',
 7: 'grapes',
 8: 'jute',
 9: 'kidneybeans',
 10: 'lentil',
 11: 'maize',
 12: 'mango',
 13: 'mothbeans',
 14: 'mungbean',
 15: 'muskmelon',
 16: 'orange',
 17: 'papaya',
 18: 'pigeonpeas',
 19: 'pomegranate',
 20: 'rice',
 21: 'watermelon'}


loaded_model=pickle.load(open('E:/resources/api-test/trained_model.sav','rb'))
def data(input_data):
    input_data_array=np.asarray(input_data) 
    reshape_array=input_data_array.reshape(1,-1)
    
    prediction=loaded_model.predict(reshape_array)
    

    if prediction[0] in kees:
        print("you should plant",targets.get(prediction[0]))
        return "you should plant " + targets.get(prediction[0])

           
def main():
    st.title('crop prediction web app')
    
    N=st.text_input("nitrogen")
    P=st.text_input("phosphorus")	
    K=st.text_input("potasiumm")
    temperature=st.text_input("temp")
    humidity=st.text_input("humidity")
    ph=st.text_input("ph")
    rainfall=st.text_input("rainfall")
    pred= ''
    if st.button('crop recommendation'):
        pred=data([N,P,K,temperature,humidity, ph, rainfall])
    st.success(pred)

if __name__=='__main__':
    main()
