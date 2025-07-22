import tempfile
import json
import os
import pandas as pd
from src.service.recruitment_training_service import RecruitmentTrainingService

def test_load_and_prepare_data():
    # Dados simulados de applicants e prospects
    applicants_data = {
        "123": {
            "infos_basicas": {"nome": "João"},
            "formacao_e_idiomas": {"nivel_ingles": "Avançado"},
            "informacoes_profissionais": {"nivel_profissional": "Sênior"},
            "cv_pt": "Experiência em Python e Machine Learning"
        }
    }

    prospects_data = {
        "123": {
            "situacao_candidado": "Encaminhado",
            "prospects": [{"vaga": "Data Scientist"}]
        }
    }

    # Criar arquivos temporários
    with tempfile.TemporaryDirectory() as temp_dir:
        applicants_path = os.path.join(temp_dir, "applicants.json")
        prospects_path = os.path.join(temp_dir, "prospects.json")

        with open(applicants_path, "w", encoding="utf-8") as f:
            json.dump(applicants_data, f)

        with open(prospects_path, "w", encoding="utf-8") as f:
            json.dump(prospects_data, f)

        # Executar o método
        service = RecruitmentTrainingService()
        df = service.load_and_prepare_data(applicants_path, prospects_path)

        # Verificações básicas
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "match_realizado" in df.columns
        assert df["match_realizado"].iloc[0] == 1
