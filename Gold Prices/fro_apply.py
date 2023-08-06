import numpy as np 
import pickle
def func():
    load_model= pickle.load(open('E:\save it\ml projects python\Gold Price\trained_model.sav','rb'))
    input_data=( 1447.160034 ,78.470001 , 15.1800 ,1.471692)
    input_data_array=np.asarray(input_data)
    reshape_array=input_data_array.reshape(1,-1)

    prediction=load_model.predict(reshape_array)
    prediction