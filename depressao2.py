import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# importação dos modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# lendo o arquivo CSV do dataset de depressão
df = pd.read_csv("",   # insira entre as aspas duplas, o caminho em que você baixou o arquivo CSV, que contem os dados
    sep=";",
    encoding="utf-8")

# seleção de colunas importantes para o treinamento
df = df[["Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", 
        "Job Satisfaction", "Sleep Duration", "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?", "Work/Study Hours",
        "Financial Stress", "Family History of Mental Illness", "Depression"]]

# remove as colunas que possuem muitos dados em uma só opção
df = df.drop(["Profession", "Work Pressure", "Job Satisfaction" ], axis=1)

# converte texto em numérico
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["pensou_em_se_matar"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df["historico_familiar"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# normalização entre 0 à 1
scaler_normal = MinMaxScaler()
df[["age_new", "Pressure_new", "nota_facul", "estudo_satisfacao", "horas_de_estudo", "estresse_financeiro"]] = scaler_normal.fit_transform(
    df[["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]])

# remove as variaveis que forma normalizadas para outro nome
df = df.drop(["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours","Have you ever had suicidal thoughts ?","Family History of Mental Illness","Financial Stress"], axis=1)

# substitui cidades com poucos registros por "Outras cidades"
df["City"] = df["City"].replace(df["City"].value_counts()[df["City"].value_counts() < 20].index, "Outras cidades")

# Agrupamento de cursos
area_estudo = {
    "MBBS": "Saúde", "MD": "Saúde", "B.Pharm": "Saúde", "M.Pharm": "Saúde",
    "B.Tech": "Engenharia/Tecnologia", "M.Tech": "Engenharia/Tecnologia", "BE": "Engenharia/Tecnologia",
    "ME": "Engenharia/Tecnologia", "B.Arch": "Engenharia/Tecnologia", "BCA": "Engenharia/Tecnologia",
    "MCA": "Engenharia/Tecnologia", "BSc": "Engenharia/Tecnologia", "MSc": "Engenharia/Tecnologia",
    "BBA": "Administração/Negócios", "MBA": "Administração/Negócios", "B.Com": "Administração/Negócios",
    "M.Com": "Administração/Negócios", "LLB": "Administração/Negócios", "LLM": "Administração/Negócios",
    "BA": "Outros", "MA": "Outros", "B.Ed": "Outros", "M.Ed": "Outros",
    "Class 12": "Outros", "PhD": "Outros", "BHM": "Outros", "MHM": "Outros", "Others": "Outros"
}
df["area_estudo"] = df["Degree"].map(area_estudo)

# ordenacao das colunas
df = df[[ 
    "Depression", "pensou_em_se_matar", "Pressure_new", "estresse_financeiro", "historico_familiar",
    "horas_de_estudo", "Sleep Duration", "estudo_satisfacao", "nota_facul", "Dietary Habits",
    "Gender", "City", "Degree", "area_estudo", "age_new"
]]

# transformar os campos categoricas em numericas
df = pd.get_dummies(df, drop_first=True)

# remove o campo depressao para realizado o treinamento
X = df.drop("Depression", axis=1)
y = df["Depression"]

# reencher as linhas vazias da coluna estresse_financeiro com a média
df["estresse_financeiro"].fillna(df["estresse_financeiro"].mean(), inplace=True)

# verifica se tem mais algum nulo, e preencho com a moda da coluna
X = X.fillna(X.mode().iloc[0])

# divisao de treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 
modelos = {
    "RandomForest": RandomForestClassifier(),
    "Bagging": BaggingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier()
}

parametros = {
    "RandomForest": {
        "n_estimators": [1453, 342, 245, 600, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [6, 5, 24],
        "min_samples_leaf": [3, 5, 4],
        "random_state": [1, 643, 14567, 5786, 2344]
    },
    "Bagging": {
        "n_estimators": [50, 100, 150, 200, 250],
        "max_samples": [0.5, 0.4, 0.75, 0.2, 1.0],
        "max_features": [0.5, 0.6, 0.75, 0.3, 1.0],
        "random_state": [0, 234, 647, 1345, 15654]
    },
    "AdaBoost": {
        "n_estimators": [504, 245],
        "learning_rate": [0.1, 0.05, 0.6],
        "random_state": [32, 545, 2353]
    },
    "GradientBoosting": {
        "n_estimators": [100, 174, 543],
        "learning_rate": [0.04, 0.05, 0.06, 0.03],
        "max_depth": [3, 2, 1],
        "random_state": [0, 23, 421, 6463]
    },
    "XGBoost": {
        "n_estimators": [100, 149, 200, 66, 14],
        "learning_rate": [0.01, 0.05, 0.02, 0.03],
        "max_depth": [7, 9, 13, 6, 8],
        "subsample": [0.6, 0.04, 0.08, 0.7],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "random_state": [66, 78, 596, 2025]
    },
    "LightGBM": {
        "n_estimators": [510, 400, 300],
        "learning_rate": [0.01, 0.009, 0.015],
        "num_leaves": [10, 22, 54],
        "boosting_type": ["gbdt", "dart"],
        "random_state": [324, 664, 2356]
    }
}


# realizacao do treinamento dos modelos
melhores_modelos = {}
resultados = []

for nome, modelo in modelos.items():
    print(f"Treinando {nome}")
    busca = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros[nome],
        n_iter=10,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    busca.fit(X_train, y_train)
    
    melhor_modelo = busca.best_estimator_
    y_pred = melhor_modelo.predict(X_test)
    
    acuracia = accuracy_score(y_test, y_pred)
    resultados.append({"Modelo": nome, "Acurácia": acuracia})
    
    print(f"Melhor modelo: {nome}: {busca.best_params_}")
    print(f"Acurácia: {acuracia:.4f}")
    print(classification_report(y_test, y_pred))
    
    melhores_modelos[nome] = melhor_modelo

    # matrix de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

# graficos de importancia dos campos por modelo
for nome, modelo in melhores_modelos.items():
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        features = X.columns
        df_imp = pd.DataFrame({"Feature": features, "Importância": importancias})
        df_imp = df_imp.sort_values(by="Importância", ascending=False).head(10)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Importância", y="Feature", data=df_imp, palette="magma")
        plt.title(f"Top 10 Features mais importantes - {nome}")
        plt.show()

