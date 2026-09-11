# Triagem Esportiva de Atletas — Saúde e Hábitos

Aplicação de Machine Learning que classifica, a partir de exames de saúde e hábitos do atleta,
se há **indício de consumo de álcool** (`DRK_YN`). O resultado apoia a **contratação de atletas**:
quanto maior o indício, mais rigorosa é a recomendação, podendo impedir a contratação até
avaliação complementar com médico do esporte.

Fluxo implementado (segue a lógica do projeto de referência):

```
Dataset → preparação dos dados → treinamento do modelo → avaliação → modelo treinado → API → aplicação
```

---

## 1. Qual dataset foi escolhido e qual problema ele representa?

**Dataset:** `smoking_driking_dataset_Ver01.csv` (dados de exames de saúde e hábitos de
~991 mil pessoas, oriundos de exames admissionais/check-up).

**Problema:** identificar, a partir dos exames de saúde de um atleta, indícios de consumo de
álcool (que comprometem desempenho, recuperação e aumentam risco de lesão). O sistema é usado
**na contratação/renovação de atletas** por clubes e assessorias esportivas, para direcionar a
anamnese e exigir avaliação médica complementar quando necessário.

## 2. Qual é a variável-alvo (target) que será prevista?

`DRK_YN` — binário gerado a partir da resposta de consumo de álcool do indivíduo.

## 3. Quais são as classes possíveis?

Duas classes:

- `Y` — indica indício de consumo de álcool (perfil de risco na triagem);
- `N` — sem indício de consumo de álcool.

## 4. Quais informações serão utilizadas como entrada do modelo?

14 features clínicas e de hábito (valores de exames + tabagismo). Foram **removidas** as
colunas pessoais/biométricas sensíveis (`sex`, `age`, `height`, `weight`, `waistline`,
`vision`, `hearing`), por não fazerem sentido — e por risco legal (LGPD) — em uma entrevista
de emprego:

| Exames (valores) | Hábito |
|---|---|
| SBP, DBP (pressão arterial) | SMK_stat_type_cd (tabagismo) |
| BLDS (glicemia) | |
| tot_chole, HDL_chole, LDL_chole, triglyceride (lipidograma) | |
| hemoglobin (hemograma) | |
| urine_protein, serum_creatinine (função renal) | |
| SGOT_AST, SGOT_ALT, gamma_GTP (função hepática) | |

## 5. Quem utilizaria essa aplicação e com qual finalidade?

**Usuários:** comissões técnicas, departamentos esportivos de clubes, assessorias de atletas e
médicos do esporte.

**Finalidade:** triagem de saúde na contratação de atletas — sinalizar quem apresenta indício
de consumo de álcool (grave para desempenho e recuperação), para direcionar a anamnese e exigir
avaliação médica complementar. Não substitui avaliação médica nem é usado como critério isolado
de contratação.

## 6. O que a aplicação fará com a classificação produzida pelo modelo?

A API responde com:

- `prediction`: classe prevista (`Y` ou `N`);
- `risk_level`: nível de risco (**SEGURO / MODERADO / CRITICO**);
- `risk_profile`: alerta gerado (mais rigoroso para atletas — atenção a partir de 35%);
- `probability`: probabilidade estimada da classe `Y`;
- `message`: mensagem orientadora (apto / avaliar com médico do esporte / impedir até avaliação).

A classificação é uma **funcionalidade útil**: transforma dados brutos de exame em um veredito
esportivo (ex.: "indício significativo → contratação impedida até avaliação complementar").

## 7. Como seria a interface ou experiência de uso dessa solução?

Uma página web (`index.html`) com formulário dos 14 exames, consumindo a API via JavaScript
`fetch()`. A comissão técnica preenche os valores, clica em **Avaliar Candidato** e recebe um
painel com veredito em 3 cores: **verde** (apto), **âmbar** (avaliação recomendada) e
**vermelho** (indício impedido até avaliação), com a probabilidade em barra visual.
Experiência simples: exames do atleta → veredito em segundos.

---

## Como executar

### Opção A — Docker (recomendado)

Pré-requisito: Docker instalado (ex.: Docker Desktop). O `model.pkl` já vem treinado
na imagem — não é necessário retreinar.

```bash
docker compose up --build
```

Acesse `http://localhost:8000`. Para parar: `docker compose down`.

### Opção B — Local (Python)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar o modelo (gera model.pkl) - pode ser pulado, o model.pkl ja existe
python train_model.py

# 3. Subir a API
uvicorn app:app --reload
```

Em ambos os casos, acesse `http://localhost:8000` para a interface, ou
`http://localhost:8000/docs` (Swagger) para testar `POST /predict` diretamente.
Na página há botões **"Teste rápido"** (perfil com/sem indício) que preenchem
os 14 campos automaticamente. O dataset CSV (109MB) é ignorado no Git/Docker
por exceder o limite do GitHub — o modelo treinado já está versionado.

---

## Resultados

- **Acurácia de teste: ~71.5%** (Pipeline `StandardScaler` + `HistGradientBoostingClassifier`,
  dataset completo, split 80/20).
- **Modelo:** `model.pkl` (serializado com `joblib`).
- **Avaliação:** acurácia em % (`accuracy_score`) no conjunto de teste.

### Comparativo de acurácia (mesma pipeline, split 80/20)

| Conjunto de features | Acurácia |
|---|---|
| Todos os dados (23 colunas originais) | **74.04%** |
| Dados atuais (14 — sem dados sensíveis) | **71.46%** |

A pequena perda (≈2.6pp) é o custo de remover colunas sensíveis (`sex`, `age`, medidas
antropométricas e exames sensoriais), mantendo o sistema adequado e legal para uso em
contratação de atletas.

## Limitações e ética

- Modelo **não é diagnóstico** médico e não deve ser usado como critério único de decisão;
- Colunas sensíveis foram removidas para reduzir risco de discriminação (LGPD);
- O sistema apoia a triagem/anamnese, cabendo a decisão final a profissionais capacitados.