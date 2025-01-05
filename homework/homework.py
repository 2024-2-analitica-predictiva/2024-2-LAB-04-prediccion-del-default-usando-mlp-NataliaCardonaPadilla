# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
import gzip
import pickle
import json
from sklearn.feature_selection import SelectKBest, f_classif

# Cargar los datos
train_data = pd.read_csv("files/input/train_data.csv.zip")
test_data = pd.read_csv("files/input/test_data.csv.zip")


# Paso 1: Limpiar los datasets
def clean_data(df):
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)  # Eliminar registros con información faltante
    df["EDUCATION"] = df["EDUCATION"].apply(
        lambda x: 4 if x > 4 else x
    )  # Agrupar EDUCATION > 4 en "others"
    return df


train_data = clean_data(train_data)
test_data = clean_data(test_data)

# Dividir en X (características) e y (etiqueta)
X_train = train_data.drop(columns="default")
y_train = train_data["default"]
X_test = test_data.drop(columns="default")
y_test = test_data["default"]

# Paso 2: Crear el pipeline
# Preprocesamiento
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Definir el pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Crear el pipeline con PCA, selección de características y red neuronal
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA()),
        ("select_kbest", SelectKBest(f_classif)),
        ("classifier", MLPClassifier(max_iter=1000)),
    ]
)

# Paso 3: Optimización de hiperparámetros con GridSearchCV
param_grid = {
    "pca__n_components": [10, 20, 30],
    "select_kbest__k": [5, 10, 15],
    "classifier__hidden_layer_sizes": [(100,), (50, 50)],
    "classifier__alpha": [0.0001, 0.001],
}

grid_search = GridSearchCV(model, param_grid, cv=10, scoring="balanced_accuracy")
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Paso 4: Guardar el modelo entrenado
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(best_model, f)


# Paso 5: Calcular las métricas y guardar en un archivo JSON
def calculate_metrics(model, X_train, y_train, X_test, y_test):
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = []

    # Métricas para el conjunto de entrenamiento
    metrics.append(
        {
            "dataset": "train",
            "precision": precision_score(y_train, y_train_pred),
            "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "f1_score": f1_score(y_train, y_train_pred),
        }
    )

    # Métricas para el conjunto de prueba
    metrics.append(
        {
            "dataset": "test",
            "precision": precision_score(y_test, y_test_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred),
        }
    )

    return metrics


metrics = calculate_metrics(best_model, X_train, y_train, X_test, y_test)

# Guardar las métricas en un archivo JSON
with open("files/output/metrics.json", "w") as f:
    for metric in metrics:
        json.dump(metric, f)
        f.write("\n")


# Paso 6: Calcular las matrices de confusión y guardarlas
def calculate_confusion_matrices(model, X_train, y_train, X_test, y_test):
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    confusion_matrices = [
        {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {"predicted_0": cm_train[0][0], "predicted_1": cm_train[0][1]},
            "true_1": {"predicted_0": cm_train[1][0], "predicted_1": cm_train[1][1]},
        },
        {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {"predicted_0": cm_test[0][0], "predicted_1": cm_test[0][1]},
            "true_1": {"predicted_0": cm_test[1][0], "predicted_1": cm_test[1][1]},
        },
    ]

    return confusion_matrices


confusion_matrices = calculate_confusion_matrices(
    best_model, X_train, y_train, X_test, y_test
)

# Guardar las matrices de confusión en el archivo JSON
with open("files/output/metrics.json", "a") as f:
    for cm in confusion_matrices:
        json.dump(cm, f)
        f.write("\n")
