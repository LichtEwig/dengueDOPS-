import uvicorn
from fastapi import FastAPI
import joblib
from dengue import MonthData
import numpy as np


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


app = FastAPI()
model = load_model("dengue_lstm_model.h5")
#joblib_in = open("modelo2.joblib","rb")
#model = joblib.load(joblib_in)

@app.get('/')
def index():
    return {'message': 'Cars Recommender ML API'}

@app.post('/car/predict')
def predict_car_type(data: dict):
    mes1 = data['month1']
    mes2 = data['month2']
    mes3 = data['month3']
    mes4 = data['month4']
    mes5 = data['month5']
    mes6 = data['month6']
    mes7 = data['month7']
    mes8 = data['month8']
    mes9 = data['month9']
    mes10 = data['month10']
    mes11 = data['month11']
    mes12 = data['month12']

    data_array = np.array([mes1, mes2, mes3, mes4, mes5, mes6, mes7, mes8, mes9, mes10, mes11, mes12])
    data_array = data_array.reshape(1, 12)
    prediction = model.predict(data_array)
    
    print("Type of prediction:", type(prediction))
    print("Shape of prediction:", prediction.shape)
    print("Prediction:", prediction)
    
    # Try to convert the prediction to a simple Python type
    try:
        if prediction.shape == (1, 1):
            prediction_value = float(prediction[0][0])
        else:
            prediction_value = prediction.flatten().tolist()
    except:
        prediction_value = "Unable to process prediction"
    
    return {
        'prediction': prediction_value
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)