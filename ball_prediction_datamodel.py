from pydantic import BaseModel

class ball_prediction(BaseModel):
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