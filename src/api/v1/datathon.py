from fastapi import APIRouter
from service import loadxxxxxxx

router = APIRouter()

@router.get('/predict')
async def get_predict():
    return 'Predict'
