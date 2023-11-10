#import library
import uvicorn
from fastapi import FastAPI
from ball_prediction_datamodel import ball_prediction
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#create the app object
app=FastAPI()
pickle_model=open("ball_prediction_rfc.pkl","rb")
classifier=pickle.load(pickle_model)

#default route
@app.get('/')
def index():
    return{"message":"Welcome to the ball prediction API. Here, we will predict the runs on each ball."}

#Prediction Function, return the predicted result in JSON
@app.post('/predict')
def predict(ball_data:ball_prediction):

    le = LabelEncoder()
    
    input_data = np.array([[
        ball_data.innings, ball_data.ball,
        ball_data.extras, ball_data.wides, ball_data.noballs,
        ball_data.byes, ball_data.legbyes, ball_data.penalty,
        le.fit_transform([ball_data.venue])[0],
        le.fit_transform([ball_data.batting_team])[0],
        le.fit_transform([ball_data.bowling_team])[0],
        le.fit_transform([ball_data.striker])[0],
        le.fit_transform([ball_data.non_striker])[0],
        le.fit_transform([ball_data.bowler])[0],
        le.fit_transform([ball_data.wicket_type])[0],
        le.fit_transform([ball_data.player_dismissed])[0],
    ]])

    predicted_runs = classifier.predict(input_data)

    predicted_runs = int(round(predicted_runs[0]))

    return {"predicted_runs": predicted_runs}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload
