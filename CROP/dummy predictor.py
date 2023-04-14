import numpy as np
import pickle 

def abikee():
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

    #loading the saved model
    loaded_model=pickle.load(open('E:/resources/api-test/trained_model.sav','rb'))
    input_data=(85,58,41,21.770462,80.319644,7.038096,226.655537)
    input_data_array=np.asarray(input_data)
    reshape_array=input_data_array.reshape(1,-1)
        
    prediction=loaded_model.predict(reshape_array)
        

    if prediction[0] in kees:
        print("you should plant",targets.get(prediction[0])) 
    return "you should plant " + targets.get(prediction[0])
