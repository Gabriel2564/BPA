# Importar librerías necesarias
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 4.1 Modelización

## 4.1.1 Árbol de Decisión
# Configuración del modelo de Árbol de Decisión
X = file_path[['Goles', 'Asistencias', 'Edad', 'Altura', 'Rating', 'Posición Auxiliar']]  # Variables predictoras
y = file_path['Valor Mercado']  # Variable objetivo (valor del mercado del jugador)

# Dividir el conjunto de datos entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Árbol de Decisión
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred_dt = dt_model.predict(X_test)
print(f"Accuracy del Árbol de Decisión: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))

# Matriz de confusión del Árbol de Decisión
print("Matriz de Confusión (Árbol de Decisión):")
print(confusion_matrix(y_test, y_pred_dt))

# 4.1.2 Random Forest
# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred_rf = rf_model.predict(X_test)
print(f"Accuracy del Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf))

# Matriz de confusión del Random Forest
print("Matriz de Confusión (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# 4.2 Optimización

## Selección de características usando Random Forest
from sklearn.feature_selection import SelectFromModel

# Usar el modelo Random Forest para seleccionar las características más importantes
sfm = SelectFromModel(rf_model, threshold=0.1)  # Establecer el umbral
sfm.fit(X_train, y_train)

# Ver qué características se seleccionaron
selected_features = X.columns[sfm.get_support()]
print(f"Características seleccionadas por Random Forest: {selected_features}")

# Transformar el conjunto de datos para mantener solo las características seleccionadas
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Re-entrenar el modelo de Random Forest con las características seleccionadas
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Evaluar el modelo con características seleccionadas
y_pred_selected = rf_model_selected.predict(X_test_selected)
print(f"Accuracy del Random Forest con características seleccionadas: {accuracy_score(y_test, y_pred_selected):.2f}")
print(classification_report(y_test, y_pred_selected))

# Matriz de confusión con las características seleccionadas
print("Matriz de Confusión (Random Forest con características seleccionadas):")
print(confusion_matrix(y_test, y_pred_selected))

