import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

class RecruitmentTrainingService:
    def __init__(self, experiment_name="Decision-Recruitment"):
        self.experiment_name = experiment_name

    def load_and_prepare_data(self, applicants_path, prospects_path):
        with open(applicants_path, "r", encoding="utf-8") as f:
            applicants_raw = json.load(f)

        applicants_list = []
        for codigo, dados in applicants_raw.items():
            infos_basicas = dados.get("infos_basicas", {})
            formacao_idiomas = dados.get("formacao_e_idiomas", {})
            infos_profissionais = dados.get("informacoes_profissionais", {})
            cv_pt = dados.get("cv_pt", "")
            row = {}
            row.update(infos_basicas)
            row.update(formacao_idiomas)
            row.update(infos_profissionais)
            row["cv_pt"] = cv_pt
            row["codigo_profissional"] = codigo
            applicants_list.append(row)
        applicants = pd.DataFrame(applicants_list)

        with open(prospects_path, "r", encoding="utf-8") as f:
            prospects_raw = json.load(f)

        if isinstance(prospects_raw, dict):
            prospects_list = []
            for codigo, dados in prospects_raw.items():
                dados["codigo"] = codigo
                prospects_list.append(dados)
            prospects = pd.DataFrame(prospects_list)
        else:
            prospects = pd.DataFrame(prospects_raw)

        df = applicants.merge(prospects, left_on="codigo_profissional", right_on="codigo", how="inner")
        df_exp = df.explode("prospects").reset_index(drop=True)
        df_exp = pd.concat([df_exp.drop(columns=["prospects"]), df_exp["prospects"].apply(pd.Series)], axis=1)

        df_exp["match_realizado"] = df_exp["situacao_candidado"].apply(lambda x: 1 if "Encaminhado" in str(x) else 0)
        return df_exp

    def preprocess_features(self, df):
        for col in ["nivel_ingles", "nivel_espanhol", "nivel_profissional"]:
            df[col] = df[col].fillna("None")

        le_ingles = LabelEncoder()
        le_espanhol = LabelEncoder()
        le_nivel = LabelEncoder()

        df["nivel_ingles_enc"] = le_ingles.fit_transform(df["nivel_ingles"])
        df["nivel_espanhol_enc"] = le_espanhol.fit_transform(df["nivel_espanhol"])
        df["nivel_profissional_enc"] = le_nivel.fit_transform(df["nivel_profissional"])

        tfidf = TfidfVectorizer(max_features=300)
        x_texto = tfidf.fit_transform(df["cv_pt"])

        x_meta = df[["nivel_ingles_enc", "nivel_espanhol_enc", "nivel_profissional_enc"]].values
        X = hstack([x_texto, x_meta])
        y = df["match_realizado"]

        encoders = {
            "le_ingles": le_ingles,
            "le_espanhol": le_espanhol,
            "le_nivel": le_nivel,
            "tfidf": tfidf
        }

        return X, y, encoders

    def train_and_log_model(self, x_train, y_train, x_test, y_test, encoders, n_estimators, random_state):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(x_train, y_train)

            preds = model.predict(x_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)

            #salvar metrica para monitoramento
            self.salvar_classification_report(report)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            # Salvar modelo e encoders localmente
            joblib.dump(model, "model.pkl")
            for name, enc in encoders.items():
                joblib.dump(enc, f"{name}.pkl")
                mlflow.log_artifact(f"{name}.pkl")

            return {
                "accuracy": acc,
                "classification_report": report
            }
        
    def salvar_classification_report(report_dict, path='.', filename='current.csv'):
        current_path = os.path.join(path, filename)
        reference_path = os.path.join(path, 'reference.csv')

        new_report_df = pd.DataFrame(report_dict).transpose()

        if not os.path.exists(current_path):
            new_report_df.to_csv(current_path, index=True)
            new_report_df.to_csv(reference_path, index=True)
            print("Primeira execução. Salvando current.csv e reference.csv.")
        else:
            current_df = pd.read_csv(current_path, index_col=0)
            current_df.to_csv(reference_path, index=True)
            new_report_df.to_csv(current_path, index=True)
            print("Atualização. current.csv atualizado e versão anterior salva em reference.csv.")

    def run_pipeline(self, applicants_path, prospects_path, n_estimators, random_state):
        df = self.load_and_prepare_data(applicants_path, prospects_path)
        X, y, encoders = self.preprocess_features(df)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return self.train_and_log_model(x_train, y_train, x_test, y_test, encoders, n_estimators, random_state)

