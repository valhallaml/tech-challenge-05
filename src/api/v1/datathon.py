from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from service import loadxxxxxxx
from service.monitoring import generate_drift_report

router = APIRouter()

@router.get('/predict')
async def get_predict():
    return 'Predict'

@router.get('/monitoring', response_class=HTMLResponse)
def get_monitoring():
    html = generate_drift_report()
    return HTMLResponse(content=html, status_code=200)
