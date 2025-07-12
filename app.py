import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE  # Para manejar clases desbalanceadas

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

# Escalar los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Comprobar si hay más de una clase en y_train
if len(np.unique(y_train)) > 1:
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    st.write("SMOTE aplicado.")
else:
    X_train_res, y_train_res = X_train, y_train
    st.write("SMOTE no aplicado, solo una clase en el entrenamiento.")

# Crear y entrenar el modelo de Árbol de Decisión con profundidad limitada
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)  # Aumentar la profundidad para capturar más complejidad
dt_model.fit(X_train_res, y_train_res)

# Realizar predicciones y evaluar el modelo
y_pred_dt = dt_model.predict(X_test)
st.write(f"Accuracy del Árbol de Decisión: {accuracy_score(y_test, y_pred_dt):.2f}")
st.write(classification_report(y_test, y_pred_dt))

# Matriz de confusión del Árbol de Decisión
st.write("Matriz de Confusión (Árbol de Decisión):")
st.write(confusion_matrix(y_test, y_pred_dt))

## 4.1.2 Random Forest
# Crear y entrenar el modelo de Random Forest con más árboles
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')  # Aumentar número de árboles
rf_model.fit(X_train_res, y_train_res)

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
rf_model_selected = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')  # Reducir el número de árboles
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
player1_index = st.number_input('Ingrese el índice del primer jugador', min_value=0, max_value=len(df)-1)
player2_index = st.number_input('Ingrese el índice del segundo jugador', min_value=0, max_value=len(df)-1)

def compare_players(player1_index, player2_index, dataframe):
    player1_features = dataframe.loc[player1_index]
    player2_features = dataframe.loc[player2_index]
    
    # Extraer los nombres de los jugadores
    player1_name = player1_features['jugador']  # Asume que la columna de nombres se llama 'jugador'
    player2_name = player2_features['jugador']  # Asume que la columna de nombres se llama 'jugador'
    
    # Mostrar las características de los jugadores seleccionados
    st.write(f"Características del Jugador 1 ({player1_name} - Índice: {player1_index}):")
    st.write(player1_features)
    st.write(f"Características del Jugador 2 ({player2_name} - Índice: {player2_index}):")
    st.write(player2_features)
    
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

if player1_index is not None and player2_index is not None:
    result = compare_players(player1_index, player2_index, df)
    st.write(result)
