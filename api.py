import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import random as rn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datamodels import *

app=FastAPI()
toss_decision=open("pickle_files/toss_decision.pkl","rb")
toss_decision_model=pickle.load(toss_decision)

file_inning1=open("pickle_files/runs_inning_1.pkl","rb")
inning1_model=pickle.load(file_inning1)

file_inning2=open("pickle_files/runs_inning_2.pkl","rb")
inning2_model=pickle.load(file_inning2)

file_over=open("pickle_files/overs.pkl","rb")
overs_model=pickle.load(file_over)

file_player=open("pickle_files/runs_wickets_prediction.pkl","rb")
player_model=pickle.load(file_player)

file_partner=open("pickle_files/run_partnership.pkl","rb")
partner_model=pickle.load(file_partner)

file_run_ball=open("pickle_files/ball_prediction_rfc.pkl","rb")
run_ball_model=pickle.load(file_run_ball)

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
    df=pd.read_csv('csv_files/matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)
    
    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    df=pd.read_csv('csv_files/upcoming_matches.csv')
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

    feature=list(df.columns)
    number_of_features=df.shape[1]
    for i in range(0,number_of_features):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df.loc[j, feature_name] = new_value

    toss_prediction = toss_decision_model.predict(df)

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

    df_points_table=pd.read_csv('csv_files/points_table.csv')
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

    df_final_points_table['NRR']=NRR
    df_final_points_table=df_final_points_table.sort_values(by=['points','NRR'],ascending=False)

    return {
        'team': list(df_final_points_table['team']),
        'points':list(df_final_points_table['points']),
        'NRR':NRR
    }

@app.post('/finalist')
def predict_finalist(top4:finalist_data):
    top4=list(top4)
    top4=top4[0][1]

    if len(top4)!=4:
        return {"message":"Must enter a list having 4 teams"}

    df=pd.read_csv('csv_files/matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)
    
    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    df=pd.read_csv('csv_files/upcoming_matches.csv')
    number_of_matches=2
    toss_winner=[]
    toss_losser=[]
    for i in range(0,number_of_matches):
        if(rn.uniform(0,1)<0.5):
            toss_winner.append(top4[i])
            toss_losser.append(top4[3-i])
        else:
            toss_winner.append(top4[3-i])
            toss_losser.append(top4[i])

    # df['toss_winner']=toss_winner

    le=LabelEncoder()
    for col in df.columns:
        df[col]=le.fit_transform(df[col])

    team1=[top4[0],top4[3]]
    team2=[top4[1],top4[2]]
    venue=['Eden Gardens','Wankhede Stadium']
    df=pd.DataFrame()
    df['team1']=team1
    df['team2']=team2
    df['venue']=venue
    df['toss_winner']=toss_winner
    feature=list(df.columns)
    number_of_features=df.shape[1]
    for i in range(0,number_of_features):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df.loc[j, feature_name] = new_value

    toss_prediction = toss_decision_model.predict(df)

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

    finalist=[]
    if inning1_pred[0]>inning2_pred[0]:
        finalist.append(batting_team1[0])
    else:
        finalist.append(batting_team2[0])

    if inning1_pred[1]>inning2_pred[1]:
        finalist.append(batting_team1[1])
    else:
        finalist.append(batting_team2[1])
    
    return{
        'finalist':finalist,
    }

@app.post('/playing-11')
def predict_playing11(finalist:playing11_data):
    team=list(finalist)
    final_team=team[0][1]

    if len(final_team)!=2:
        return {"message":"Must enter a list having 2 teams"}

    df=pd.read_csv('csv_files/playerwise_df.csv')
    df.drop(['match_id', 'season', 'start_date'],axis='columns',inplace=True)

    le = LabelEncoder()
    mapping = {}

    for column in df.drop(['total_runs','total_wickets'],axis='columns').columns:
        df[column] = le.fit_transform(df[column])
        mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    df_deliveries = pd.read_csv('csv_files/deliveries.csv')
    all_players = set()

    all_players.update(df_deliveries['striker'].unique())
    all_players.update(df_deliveries['non_striker'].unique())
    all_players.update(df_deliveries['bowler'].unique())

    team_players = {team: {'striker': set(), 'non_striker': set(), 'bowler': set()} for team in df_deliveries['batting_team'].unique()}

    for _, row in df_deliveries.iterrows():
        team = row['batting_team']
        team_players[team]['striker'].add(row['striker'])
        team_players[team]['non_striker'].add(row['non_striker'])
        team_players[team]['bowler'].add(row['bowler'])

    team_players = {team: set() for team in df_deliveries['batting_team'].unique()}

    for _, row in df_deliveries.iterrows():
        team = row['batting_team']
        team_players[team].add(row['striker'])
        team_players[team].add(row['non_striker'])
        
    for _, row in df_deliveries.iterrows():
        team = row['bowling_team']
        team_players[team].add(row['bowler'])

    team1_players=list(team_players[final_team[0]])
    team2_players=list(team_players[final_team[1]])

    team1=final_team[0]
    team2=final_team[1]

    df_team1=pd.DataFrame()
    df_team1['venue']=[7 for i in range(0,len(team1_players))]
    df_team1['team']=[team1 for i in range(0,df_team1.shape[0])]
    df_team1['opposing_team']=[team2 for i in range(0,df_team1.shape[0])]
    df_team1['player']=team1_players

    number_of_players=df_team1.shape[0]
    feature=list(df_team1.columns)
    number_of_features=df_team1.shape[1]
    for i in range(1,number_of_features):
        for j in range(0,number_of_players):
            feature_name = feature[i]
            original_value = df_team1.loc[j, feature_name]
            new_value = mapping[feature_name][original_value]
            df_team1.loc[j, feature_name] = int(new_value)

    df_team1=df_team1.astype('int')

    player_team1=player_model.predict(df_team1)

    df_team2=pd.DataFrame()
    df_team2['venue']=[7 for i in range(0,len(team2_players))]
    df_team2['team']=[team2 for i in range(0,df_team2.shape[0])]
    df_team2['opposing_team']=[team1 for i in range(0,df_team2.shape[0])]
    df_team2['player']=team2_players

    number_of_players=df_team2.shape[0]
    feature=list(df_team2.columns)
    number_of_features=df_team2.shape[1]
    for i in range(1,number_of_features):
        for j in range(0,number_of_players):
            feature_name = feature[i]
            original_value = df_team2.loc[j, feature_name]
            new_value = mapping[feature_name][original_value]
            df_team2.loc[j, feature_name] = new_value

    df_team2=df_team2.astype('int')

    player_team2=player_model.predict(df_team2)

    final_team1=pd.DataFrame()
    final_team1['player']=team1_players
    run=[]
    for val in player_team1:
        run.append(val[0])
    final_team1['run']=run

    wicket=[]
    for val in player_team1:
        wicket.append(val[1])
    final_team1['wicket']=wicket

    final_team2=pd.DataFrame()
    final_team2['player']=team2_players
    run=[]
    for val in player_team2:
        run.append(val[0])
    final_team2['run']=run

    wicket=[]
    for val in player_team2:
        wicket.append(val[1])
    final_team2['wicket']=wicket

    final_team1=final_team1.sort_values(by=['run','wicket'],ascending=False)
    final_team2=final_team2.sort_values(by=['run','wicket'],ascending=False)

    team1_playing_11=[]
    team2_playing_11=[]
    for i in range(0,12):
        team1_playing_11.append(final_team1.iloc[i]['player'])
        team2_playing_11.append(final_team2.iloc[i]['player'])

    return {
        team1:team1_playing_11,
        team2:team2_playing_11
    }

@app.post('/winner')
def predict_winner(finalist:winner_data):
    finalist=list(finalist)
    finalist_team=finalist[0][1]

    if len(finalist_team)!=2:
        return {"message":"Must enter a list having 2 teams"}

    df=pd.read_csv('csv_files/matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)
    
    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    df=pd.read_csv('csv_files/upcoming_matches.csv')
    number_of_matches=1
    toss_winner=[]
    toss_losser=[]
    for i in range(0,number_of_matches):
        if(rn.uniform(0,1)<0.5):
            toss_winner.append(finalist_team[i])
            toss_losser.append(finalist_team[1-i])
        else:
            toss_winner.append(finalist_team[1-i])
            toss_losser.append(finalist_team[i])

    le=LabelEncoder()
    for col in df.columns:
        df[col]=le.fit_transform(df[col])

    team1=[finalist_team[0]]
    team2=[finalist_team[1]]
    venue=['Narendra Modi Stadium']
    df=pd.DataFrame()
    df['team1']=team1
    df['team2']=team2
    df['venue']=venue
    df['toss_winner']=toss_winner
    feature=list(df.columns)
    number_of_features=df.shape[1]
    for i in range(0,number_of_features):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df.loc[j, feature_name]
            new_value = toss_mapping[feature_name][original_value]
            df.loc[j, feature_name] = new_value

    toss_prediction = toss_decision_model.predict(df)

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

    if inning1_pred[0]>inning2_pred[0]:
        winner=batting_team1[0]
    else:
        winner=batting_team2[0]

    return{
        'winner':winner,
    }

@app.post('/partnership')
def predict_partnership_runs(data:partner_data):
    data=dict(data)
    data_dict={}
    data_dict['venue']=data['venue']
    data_dict['batting_team']=data['batting_team']
    data_dict['bowling_team']=data['bowling_team']
    data_dict['striker']=data['striker']
    data_dict['non_striker']=data['non_striker']

    le = LabelEncoder()
    mapping = {}

    df_deliveries=pd.read_csv('csv_files/deliveries.csv')

    df_deliveries = df_deliveries.drop(['innings','season','start_date', 'ball','bowler', 'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
       'penalty','other_wicket_type','other_player_dismissed','player_dismissed','match_id', 'wicket_type'],axis='columns')

    cat = [col for col in df_deliveries if df_deliveries[col].dtype == 'object']

    for column in cat:
        df_deliveries[column] = le.fit_transform(df_deliveries[column])
        mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    feature=list(df_deliveries.columns)
    number_of_features=df_deliveries.shape[1]
    for i in range(0,number_of_features):
        feature_name = feature[i]
        original_value = data_dict[feature_name]
        new_value = mapping[feature_name][original_value]
        data_dict[feature_name] = new_value

    df_input=pd.DataFrame()
    # tmp=[[data_dict['venue'],data_dict['batting_team'],data_dict['bowling_team'],data_dict['striker'],data_dict['non_striker']],[data_dict['venue'],data_dict['batting_team'],data_dict['bowling_team'],data_dict['striker'],data_dict['non_striker']]]

    df_input['venue']=[data_dict['venue']]
    df_input['batting_team']=[data_dict['batting_team']]
    df_input['bowling_team']=[data_dict['bowling_team']]
    df_input['striker']=[data_dict['striker']]
    df_input['non_striker']=[data_dict['non_striker']]

    runs=partner_model.predict(df_input)

    return{
        "partnership_runs":runs[0]
    }

@app.post('/run-on-ball')
def predict_runs_on_ball(ball_data:ball_prediction_data):
    
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

    predicted_runs = run_ball_model.predict(input_data)

    predicted_runs = int(round(predicted_runs[0]))

    return {"predicted_runs": predicted_runs}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

#Command to run API server   
#python -m uvicorn main:app --reload

