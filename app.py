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

# Si el dataset es muy grande, tomar una muestra aleatoria (10%)
df = df.sample(frac=0.1, random_state=42)

# Restablecer el índice
df = df.reset_index(drop=True)

# Mostrar las primeras filas del archivo
st.write("Vista previa de los datos:")
st.write(df.head())

# 4.1 Modelización

## 4.1.1 Árbol de Decisión
# Configuración del modelo de Árbol de Decisión
X = df[['goles', 'asistencias', 'edad', 'altura', 'rating', 'posición auxiliar']]  # Variables predictoras
y = df['valor']  # Variable objetivo (valor del mercado del jugador)

# Dividir el conjunto de datos entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Árbol de Decisión con profundidad limitada
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)  # Limitar la profundidad para optimizar el uso de memoria
dt_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred_dt = dt_model.predict(X_test)
st.write(f"Accuracy del Árbol de Decisión: {accuracy_score(y_test, y_pred_dt):.2f}")
st.write(classification_report(y_test, y_pred_dt))

# Matriz de confusión del Árbol de Decisión
st.write("Matriz de Confusión (Árbol de Decisión):")
st.write(confusion_matrix(y_test, y_pred_dt))

## 4.1.2 Random Forest
# Crear y entrenar el modelo de Random Forest con menos árboles para optimizar memoria
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reducir número de árboles
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
rf_model_selected = RandomForestClassifier(n_estimators=50, random_state=42)  # Reducir el número de árboles
rf_model_selected.fit(X_train_selected, y_train)

# Evaluar el modelo con características seleccionadas
y_pred_selected = rf_model_selected.predict(X_test_selected)
st.write(f"Accuracy del Random Forest con características seleccionadas: {accuracy_score(y_test, y_pred_selected):.2f}")
st.write(classification_report(y_test, y_pred_selected))

# Matriz de confusión con las características seleccionadas
st.write("Matriz de Confusión (Random Forest con características seleccionadas):")
st.write(confusion_matrix(y_test, y_pred_selected))

# Comparación de jugadores (Función de recomendación)
st.write("Comparar dos jugadores:")

# Cambié la entrada a nombre del jugador
player1_name = st.text_input('Ingrese el nombre del primer jugador')
player2_name = st.text_input('Ingrese el nombre del segundo jugador')

def compare_players(player1_name, player2_name, dataframe):
    # Buscar los jugadores por nombre
    player1_features = dataframe[dataframe['jugador'] == player1_name]
    player2_features = dataframe[dataframe['jugador'] == player2_name]
    
    # Verificar si ambos jugadores existen
    if player1_features.empty or player2_features.empty:
        return "Uno o ambos jugadores no se encuentran en el conjunto de datos."
    
    player1_features = player1_features.iloc[0]
    player2_features = player2_features.iloc[0]
    
    score1 = 0
    score2 = 0

    weights = {
        'rating': 0.3, 'edad': 0.1, 'pt': 0.1, 'posición auxiliar': 0.05,
        'altura': 0.05, 'asistencias': 0.15, 'pj': 0.05, 'goles': 0.15, 'ta': -0.05
    }

    # Comparación de cada característica
    comparison_details = []
    for feature, weight in weights.items():
        if weight > 0:
            if player1_features[feature] > player2_features[feature]:
                score1 += weight
                comparison_details.append(f"{feature}: {player1_name} tiene mayor valor ({player1_features[feature]}) que {player2_name} ({player2_features[feature]})")
            elif player2_features[feature] > player1_features[feature]:
                score2 += weight
                comparison_details.append(f"{feature}: {player2_name} tiene mayor valor ({player2_features[feature]}) que {player1_name} ({player1_features[feature]})")
            else:
                comparison_details.append(f"{feature}: Ambos jugadores tienen el mismo valor ({player1_features[feature]})")
        elif weight < 0:
            if player1_features[feature] < player2_features[feature]:
                score1 += abs(weight)
                comparison_details.append(f"{feature}: {player1_name} tiene menor valor ({player1_features[feature]}) que {player2_name} ({player2_features[feature]})")
            elif player2_features[feature] < player1_features[feature]:
                score2 += abs(weight)
                comparison_details.append(f"{feature}: {player2_name} tiene menor valor ({player2_features[feature]}) que {player1_name} ({player1_features[feature]})")
            else:
                comparison_details.append(f"{feature}: Ambos jugadores tienen el mismo valor ({player1_features[feature]})")

    # Mostrar los detalles de la comparación
    st.write("Detalles de la comparación:")
    for detail in comparison_details:
        st.write(detail)

    # Determinar cuál jugador es mejor
    if score1 > score2:
        return f"{player1_name} es mejor que {player2_name}."
    elif score2 > score1:
        return f"{player2_name} es mejor que {player1_name}."
    else:
        return f"Los jugadores {player1_name} y {player2_name} son comparables."

if player1_name and player2_name:
    result = compare_players(player1_name, player2_name, df)
    st.write(result)
