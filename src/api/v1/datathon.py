from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from src.service.recruitment_training_service import RecruitmentTrainingService
from service.monitoring import generate_drift_report
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

router = APIRouter()


clientRecruitmentTrainingService = RecruitmentTrainingService()

# Carregar artefatos
model = joblib.load("model_applicants.pkl")
tfidf = joblib.load("tfidf_applicants.pkl")
le_ingles = joblib.load("le_ingles.pkl")
le_espanhol = joblib.load("le_espanhol.pkl")
le_nivel = joblib.load("le_nivel.pkl")
class Candidate(BaseModel):
    cv_pt: str
    nivel_ingles: str
    nivel_espanhol: str
    nivel_profissional: str

# --- Input do endpoint de treino ---
class TrainInput(BaseModel):
    applicants_path: str
    prospects_path: str

@app.post("/predict")
def predict(candidate: Candidate):
    # Pré-processar entrada
    cv_vector = tfidf.transform([candidate.cv_pt])
    ni_enc = le_ingles.transform([candidate.nivel_ingles])
    ne_enc = le_espanhol.transform([candidate.nivel_espanhol])
    np_enc = le_nivel.transform([candidate.nivel_profissional])

    meta_features = [[ni_enc[0], ne_enc[0], np_enc[0]]]
    X = hstack([cv_vector, meta_features])

    pred = model.predict(X)
    proba = model.predict_proba(X)

    return {
        "prediction": int(pred[0]),
        "probability": proba[0].tolist()
    }

@app.post("/train")
def treinar_modelo(data: TrainInput):
    service = RecruitmentTrainingService()
    resultado = service.run_pipeline(data.applicants_path, data.prospects_path)
    return {
        "status": "Treinamento concluído com sucesso.",
        "metrics": resultado
    }

@router.get('/monitoring', response_class=HTMLResponse)
def get_monitoring():
    html = generate_drift_report()
    return HTMLResponse(content=html, status_code=200)
