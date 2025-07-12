import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# Título de la aplicación
st.title('Análisis de Jugadores de Fútbol')

# URL del archivo en GitHub (URL raw del archivo Excel)
file_url = 'https://raw.githubusercontent.com/Gabriel2564/TF/b0e20aca29299e25742f16d42966c1386e325b5f/DataSet_Jugadores_Categorizado-bpa.xlsx'

# Cargar el dataset desde la URL
df = pd.read_excel(file_url)

# 4.1 Modelización

## 4.1.1 Árbol de Decisión
# Configuración del modelo de Árbol de Decisión
X = df[['goles', 'asistencias', 'edad', 'altura', 'rating', 'posición auxiliar']]  # Variables predictoras
y = df['valor']  # Variable objetivo (valor del mercado del jugador)

# Dividir el conjunto de datos entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Árbol de Decisión
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred_dt = dt_model.predict(X_test)
st.write(f"Accuracy del Árbol de Decisión: {accuracy_score(y_test, y_pred_dt):.2f}")
st.write(classification_report(y_test, y_pred_dt))

# Matriz de confusión del Árbol de Decisión
st.write("Matriz de Confusión (Árbol de Decisión):")
st.write(confusion_matrix(y_test, y_pred_dt))

## 4.1.2 Random Forest
# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred_rf = rf_model.predict(X_test)
st.write(f"Accuracy del Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
st.write(classification_report(y_test, y_pred_rf))

# Matriz de confusión del Random Forest
st.write("Matriz de Confusión (Random Forest):")
st.write(confusion_matrix(y_test, y_pred_rf))

# 4.2 Optimización

## Selección de características usando Random Forest
# Usar el modelo Random Forest para seleccionar las características más importantes
sfm = SelectFromModel(rf_model, threshold=0.1)  # Establecer el umbral
sfm.fit(X_train, y_train)

# Ver qué características se seleccionaron
selected_features = X.columns[sfm.get_support()]
st.write(f"Características seleccionadas por Random Forest: {selected_features}")

# Transformar el conjunto de datos para mantener solo las características seleccionadas
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Re-entrenar el modelo de Random Forest con las características seleccionadas
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Evaluar el modelo con características seleccionadas
y_pred_selected = rf_model_selected.predict(X_test_selected)
st.write(f"Accuracy del Random Forest con características seleccionadas: {accuracy_score(y_test, y_pred_selected):.2f}")
st.write(classification_report(y_test, y_pred_selected))

# Matriz de confusión con las características seleccionadas
st.write("Matriz de Confusión (Random Forest con características seleccionadas):")
st.write(confusion_matrix(y_test, y_pred_selected))
