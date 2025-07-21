from pydantic import BaseModel

class Candidate(BaseModel):
    cv_pt: str
    nivel_ingles: str
    nivel_espanhol: str
    nivel_profissional: str
