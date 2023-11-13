from pydantic import BaseModel

class finalist_data(BaseModel):
    team:list

class playing11_data(BaseModel):
    team:list

class winner_data(BaseModel):
    team:list

class partner_data(BaseModel):
    venue:str
    batting_team:str
    bowling_team:str
    striker:str
    non_striker:str
    
class ball_prediction_data(BaseModel):
    venue: str
    innings: int
    ball: float
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    extras: int
    wides: float
    noballs: float
    byes: float
    legbyes: float
    penalty: float
    wicket_type: str
    player_dismissed: str