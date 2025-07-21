from pydantic import BaseModel

class TrainRequest(BaseModel):
    n_estimators: int
    random_state: int
