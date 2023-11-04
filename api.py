import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import random as rn
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

app=FastAPI()
toss_decision=open("toss_decision.pkl","rb")
toss_decision_model=pickle.load(toss_decision)

#default route
@app.get('/')
def index():
    return{"message":"Group 03"}

#default route
@app.get('/api-demo')
def index():
    return{"message":"This is Cricket World Cup 2023 predictor API"}

#Prediction Function, return the predicted result in JSON
@app.get('/finals')
def predict():
    df=pd.read_csv('upcoming_matches.csv')
    number_of_matches=df.shape[0]
    toss_winner=[]
    for i in range(0,number_of_matches):
        if(rn.uniform(0,1)<0.5):
            toss_winner.append(df.iloc[i]['team1'])
        else:
            toss_winner.append(df.iloc[i]['team2'])

    df['toss_winner']=toss_winner
    columns=df.columns

    le=LabelEncoder()
    for col in df.columns:
        df[col]=le.fit_transform(df[col])

    prediction = toss_decision_model.predict(df)
    if(prediction[2]==0):
        prediction="Bat"
    else:
        prediction="Field"
    return {
        'prediction': prediction,
        'toss_winner':toss_winner[2]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload

