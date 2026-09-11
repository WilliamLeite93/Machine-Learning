import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "smoking_driking_dataset_Ver01.csv")

print("[1/3] Carregando o dataset...")
df = pd.read_csv(CSV_PATH)

df['DRK_YN'] = (df['DRK_YN'] == 'Y').astype(int)

features = [
    'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride',
    'hemoglobin', 'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT',
    'gamma_GTP', 'SMK_stat_type_cd',
]
X = df[features]  
y = df['DRK_YN'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = make_pipeline(
    StandardScaler(),
    HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42))
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"[2/3] Modelo treinado! Acurácia de teste: {acc * 100:.2f}%")

joblib.dump(model, os.path.join(BASE_DIR, 'model.pkl'))
print("[3/3] Modelo salvo com sucesso em 'model.pkl'!")