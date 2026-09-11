import os
from contextlib import asynccontextmanager
from enum import Enum

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = [
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "HDL_chole",
    "LDL_chole",
    "triglyceride",
    "hemoglobin",
    "urine_protein",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
    "gamma_GTP",
    "SMK_stat_type_cd",
]

model_data = None

class Context(str, Enum):
    rh = "rh"
    sport = "sport"


class CandidateIn(BaseModel):
    SBP: float
    DBP: float
    BLDS: float
    tot_chole: float
    HDL_chole: float
    LDL_chole: float
    triglyceride: float
    hemoglobin: float
    urine_protein: float
    serum_creatinine: float
    SGOT_AST: float
    SGOT_ALT: float
    gamma_GTP: float
    SMK_stat_type_cd: float = Field(..., description="Status de tabagismo (1=Nunca, 2=Ex-fumante, 3=Fumante)")
    context: Context = Context.sport

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_data
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Arquivo de modelo '{MODEL_PATH}' não encontrado. "
            "Execute 'python train_model.py' antes de iniciar a API."
        )
    with open(MODEL_PATH, "rb") as file:
        model_data = joblib.load(file)
    print(f"Modelo carregado com sucesso. Features: {len(FEATURES)}")
    yield


app = FastAPI(title="Triagem de Atletas - Saúde e Hábitos", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model_data is not None}


def to_input_vector(payload: CandidateIn) -> list:
    """Converte o JSON recebido no vetor na ordem exata usada no treinamento."""
    values = payload.model_dump()
    return [values[feature] for feature in FEATURES]


def classify_risk(probability: float, context: Context):
    if probability < 0.35:
        level, risk = "SEGURO", False
    elif probability < 0.60:
        level, risk = "MODERADO", context == Context.sport
    else:
        level, risk = "CRITICO", True

    if context == Context.sport:
        messages = {
            "SEGURO": "Sem indício de consumo de álcool - apto para atividade esportiva.",
            "MODERADO": "Atenção: recomenda-se avaliação com médico do esporte antes da contratação.",
            "CRITICO": "Indício significativo de consumo de álcool: vetar participação até avaliação complementar.",
        }
    else:
        messages = {
            "SEGURO": f"Baixo indício de consumo de álcool (probabilidade de {probability:.1%}).",
            "MODERADO": f"Indício moderado de consumo de álcool (probabilidade de {probability:.1%}). Recomenda-se aprofundar na anamnese.",
            "CRITICO": f"Indício significativo de consumo de álcool (probabilidade de {probability:.1%}). Recomenda-se avaliação com médico do trabalho.",
        }

    return level, risk, messages[level]


@app.post("/predict")
def predict(payload: CandidateIn) -> dict:
    if model_data is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    model = model_data
    vector = to_input_vector(payload)

    predicted = int(model.predict([vector])[0])        
    probability = float(model.predict_proba([vector])[0][1]) 

    label = "Y" if predicted == 1 else "N"
    level, risk, message = classify_risk(probability, payload.context)

    return {
        "prediction": label,                
        "context": payload.context.value,
        "risk_level": level,                
        "risk_profile": risk,               
        "probability": round(probability, 4),
        "message": message,
    }