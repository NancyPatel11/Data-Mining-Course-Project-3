from pydantic import BaseModel

class finalist_data(BaseModel):
    team:list

class playing11_data(BaseModel):
    team:list

class winner_data(BaseModel):
    team:list