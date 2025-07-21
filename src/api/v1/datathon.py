from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from service.recruitment_training_service import RecruitmentTrainingService
from service.monitoring import generate_drift_report
from models import Candidate, TrainRequest

import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

router = APIRouter()

@router.post('/predict')
def predict(candidate: Candidate):

    # Carregar artefatos
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    le_ingles = joblib.load('le_ingles.pkl')
    le_espanhol = joblib.load('le_espanhol.pkl')
    le_nivel = joblib.load('le_nivel.pkl')

    # Pré-processar entrada
    cv_vector = tfidf.transform([candidate.cv_pt])
    ni_enc = le_ingles.transform([candidate.nivel_ingles])
    ne_enc = le_espanhol.transform([candidate.nivel_espanhol])
    np_enc = le_nivel.transform([candidate.nivel_profissional])

    meta_features = [[ni_enc[0], ne_enc[0], np_enc[0]]]
    x = hstack([cv_vector, meta_features])

    pred = model.predict(x)

    return {
        'prediction': int(pred[0])
    }

@router.post('/train')
def treinar_modelo(train_request: TrainRequest = TrainRequest(n_estimators=100, random_state=42)):
    service = RecruitmentTrainingService()
    resultado = service.run_pipeline('src/data/applicants.json', 'src/data/prospects.json', train_request)
    return {
        'status': 'Treinamento concluído com sucesso.',
        'metrics': resultado
    }

@router.get('/monitoring', response_class = HTMLResponse)
def get_monitoring():
    html = generate_drift_report()
    return HTMLResponse(content=html, status_code=200)
