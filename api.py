import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import random as rn
from sklearn.preprocessing import LabelEncoder

app=FastAPI()
toss_decision=open("toss_decision.pkl","rb")
toss_decision_model=pickle.load(toss_decision)

file_inning1=open("runs_inning_1.pkl","rb")
inning1_model=pickle.load(file_inning1)

file_inning2=open("runs_inning_2.pkl","rb")
inning2_model=pickle.load(file_inning2)

file_over=open("overs.pkl","rb")
overs_model=pickle.load(file_over)

#default route
@app.get('/')
def index():
    return{"message":"Group 03"}

#default route
@app.get('/api-demo')
def index():
    return{"message":"This is Cricket World Cup 2023 predictor API"}

#Prediction Function, return the predicted result in JSON
@app.get('/points-table')
def predict():
    df=pd.read_csv('matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)
    
    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    toss_mapping

    df=pd.read_csv('upcoming_matches.csv')
    number_of_matches=df.shape[0]
    toss_winner=[]
    toss_losser=[]
    for i in range(0,number_of_matches):
        if(rn.uniform(0,1)<0.5):
            toss_winner.append(df.iloc[i]['team1'])
            toss_losser.append(df.iloc[i]['team2'])
        else:
            toss_winner.append(df.iloc[i]['team2'])
            toss_losser.append(df.iloc[i]['team1'])

    df['toss_winner']=toss_winner

    # le=LabelEncoder()
    # for col in df.columns:
    #     df[col]=le.fit_transform(df[col])

    feature=list(df.columns)
    number_of_features=df.shape[1]
    for i in range(0,number_of_features):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df.loc[j, feature_name] = new_value

    toss_prediction = toss_decision_model.predict(df)
    # prediction=1 => field
    # prediction=0 => bat

    batting_team1=[]
    bowling_team1=[]
    venue_x=[]
    Total_Overs_Played=[]
    for i in range(0,number_of_matches):
        if toss_prediction[i]==0:
            batting=toss_winner[i]
            bowling=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)
        else:
            bowling=toss_winner[i]
            batting=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)

    df_inning1=pd.DataFrame()
    df_inning1['team1']=batting_team1
    df_inning1['team2']=bowling_team1
    df_inning1['venue']=venue_x
    df_inning1['Total_Overs_Played']=Total_Overs_Played

    feature=list(df_inning1.columns)
    number_of_features=df_inning1.shape[1]
    for i in range(0,number_of_features-2):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df_inning1.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df_inning1.loc[j, feature_name] = new_value

    df_inning1.rename(columns={"team1":"batting_team","team2":"bowling_team","venue":"venue_x"},inplace=True)
    inning1_pred=inning1_model.predict(df_inning1)

    df_inning1.drop('Total_Overs_Played',axis='columns',inplace=True)
    df_inning1['total_runs_per_inning_match']=list(inning1_pred)
    df_inning1['innings']=[1 for i in range(0,number_of_matches)]

    overs_pred1=overs_model.predict(df_inning1)

    for i in range(0,len(inning1_pred)):
        if overs_pred1[i]>50.0:
            inning1_pred[i]=inning1_pred[i]*50/overs_pred1[i]

    ###2
    batting_team2=[]
    bowling_team2=[]
    venue_x=[]
    Total_Overs_Played=[]
    for i in range(0,number_of_matches):
        if toss_prediction[i]==0:
            bowling=toss_winner[i]
            batting=toss_losser[i]
            batting_team2.append(batting)
            bowling_team2.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)
        else:
            batting=toss_winner[i]
            bowling=toss_losser[i]
            batting_team2.append(batting)
            bowling_team2.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)

    df_inning2=pd.DataFrame()
    df_inning2['team1']=batting_team2
    df_inning2['team2']=bowling_team2
    df_inning2['venue']=venue_x
    df_inning2['Total_Overs_Played']=Total_Overs_Played

    feature=list(df_inning2.columns)
    number_of_features=df_inning2.shape[1]
    for i in range(0,number_of_features-2):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df_inning2.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df_inning2.loc[j, feature_name] = new_value

    df_inning2.rename(columns={"team1":"batting_team","team2":"bowling_team","venue":"venue_x"},inplace=True)
    inning2_pred=inning2_model.predict(df_inning2)

    df_inning2.drop('Total_Overs_Played',axis='columns',inplace=True)
    df_inning2['total_runs_per_inning_match']=list(inning2_pred)
    df_inning2['innings']=[2 for i in range(0,number_of_matches)]

    overs_pred2=overs_model.predict(df_inning2)

    for i in range(0,len(inning2_pred)):
        if overs_pred2[i]>50.0:
            inning2_pred[i]=inning2_pred[i]*50/overs_pred2[i]

    for i in range(0,len(inning1_pred)):
        if inning1_pred[i]>inning2_pred[i]:
            overs_pred2[i]=(50.0)
        else:
            overs_pred2=50.0*(inning1_pred[i]+1)/inning2_pred[i]

    overs_for=[]
    overs_against=[]

    df_points_table=pd.read_csv('points_table.csv')
    team=list(df_points_table['Team'])
    points=list(df_points_table['Points'])
    for_=list(df_points_table['For'])
    against=list(df_points_table['Against'])

    for i in range(0,len(for_)):
        tmp=str(for_[i]).split("/")
        for_[i]=int(tmp[0])
        overs_for.append(float(tmp[1]))

    for i in range(0,len(against)):
        tmp=str(against[i]).split("/")
        against[i]=int(tmp[0])
        overs_against.append(float(tmp[1]))

    df_final_points_table=pd.DataFrame()
    df_final_points_table['team']=team
    df_final_points_table['points']=points
    df_final_points_table['for_']=for_
    df_final_points_table['against']=against
    df_final_points_table['overs_for']=overs_for
    df_final_points_table['overs_against']=overs_against

    dict_final_points={}

    i=0
    for country in df_final_points_table['team']:
        dict_final_points[country]=i
        i=i+1

    for i in range(0,len(batting_team1)):
        if inning1_pred[i]>inning2_pred[i]:
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'points']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'points']+2
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'for_']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'for_']+inning1_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'against']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'against']+inning2_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_for']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_for']+overs_pred1[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_against']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_against']+overs_pred2[i]

            df_final_points_table.loc[dict_final_points[batting_team2[i]],'for_']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'for_']+inning2_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'against']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'against']+inning1_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_for']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_for']+overs_pred2[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_against']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_against']+overs_pred1[i]
        else:
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'points']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'points']+2
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'for_']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'for_']+inning2_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'against']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'against']+inning1_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_for']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_for']+overs_pred2[i]
            df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_against']=df_final_points_table.loc[dict_final_points[batting_team2[i]],'overs_against']+overs_pred1[i]

            df_final_points_table.loc[dict_final_points[batting_team1[i]],'for_']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'for_']+inning1_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'against']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'against']+inning2_pred[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_for']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_for']+overs_pred1[i]
            df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_against']=df_final_points_table.loc[dict_final_points[batting_team1[i]],'overs_against']+overs_pred2[i]

    NRR=[]
    for i in range(0,df_final_points_table.shape[0]):
        for_runs=df_final_points_table.iloc[i]['for_']
        for_overs=df_final_points_table.iloc[i]['overs_for']
        final_for=for_runs/for_overs

        against_runs=df_final_points_table.iloc[i]['against']
        against_overs=df_final_points_table.iloc[i]['overs_against']
        final_against=against_runs/against_overs

        NRR.append(final_for-final_against)

    return {
        'team': list(df_final_points_table['team']),
        'points':list(df_final_points_table['points']),
        'NRR':NRR
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload

