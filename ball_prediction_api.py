#import library
import uvicorn
from fastapi import FastAPI
from ball_prediction_datamodel import ball_prediction
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

#create the app object
app=FastAPI()
pickle_model=open("pickle_files/ball_prediction_rfc.pkl","rb")
classifier=pickle.load(pickle_model)

#default route
@app.get('/')
def index():
    return{"message":"Welcome to the ball prediction API. Here, we will predict the runs on each ball."}

data = pd.read_csv("csv_files/ball_prediction.csv",index_col=0)
data.drop(['match_id','season','start_date','runs_off_bat'],axis=1,inplace=True)

def label_encode(data, column, le=None):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return le

# Example of label encoding for string columns
label_encoders = {}
for column in ['venue', 'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler', 'wicket_type', 'player_dismissed']:
    label_encoders[column] = label_encode(data, column)

scaler = StandardScaler()
data = scaler.fit_transform(data)

#Prediction Function, return the predicted result in JSON
@app.post('/predict')
def predict(ball_data:ball_prediction):
    
    input_data = np.array([[
        label_encoders['venue'].transform([ball_data.venue])[0],
        ball_data.innings, ball_data.ball,
        label_encoders['batting_team'].transform([ball_data.batting_team])[0],
        label_encoders['bowling_team'].transform([ball_data.bowling_team])[0],
        label_encoders['striker'].transform([ball_data.striker])[0],
        label_encoders['non_striker'].transform([ball_data.non_striker])[0],
        label_encoders['bowler'].transform([ball_data.bowler])[0],
        ball_data.extras, ball_data.wides, ball_data.noballs,
        ball_data.byes, ball_data.legbyes, ball_data.penalty,
        label_encoders['wicket_type'].transform([ball_data.wicket_type])[0],
        label_encoders['player_dismissed'].transform([ball_data.player_dismissed])[0],
    ]])

    input_data = scaler.transform(input_data)

    predicted_runs = classifier.predict(input_data)

    predicted_runs = int(round(predicted_runs[0]))

    return {"predicted_runs": predicted_runs}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload
