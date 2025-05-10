import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv('C:/Users/AhxelLuis/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PRUEBAS2/DATA/DATAPERU2CSV.csv')

# Limpiar las columnas numéricas eliminando comas y convirtiéndolas a float
columns_to_clean = ['CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']
for col in columns_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True).astype(float)

# Crear una variable objetivo ficticia (puedes definir una lógica más adelante)
df['comprar_complementario'] = (df['V_TOTAL'] > 1000).astype(int)

# Selección de características (X) y variable objetivo (y)
features = ['TIPO_VENTA', 'TIPO_CLIENTE', 'SKU', 'ARTICULO', 'CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']
X = df[features]
y = df['comprar_complementario']

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento: Codificación de variables categóricas y normalización de variables numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']),  # Normalizar características numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['TIPO_VENTA', 'TIPO_CLIENTE', 'SKU', 'ARTICULO'])  # Codificación one-hot de características categóricas
    ])

# Crear el pipeline con el clasificador RandomForest
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))  # Limitar la profundidad de los árboles
])

# Entrenamiento del modelo con validación cruzada
cv_scores = cross_val_score(model_pipeline, X, y, cv=5)  # Validación cruzada con 5 folds
print(f"Precisión media con validación cruzada: {cv_scores.mean()}")

# Entrenamiento final con todo el conjunto de datos de entrenamiento
model_pipeline.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = model_pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Imprimir la precisión
print(f'Precisión del modelo en los datos de prueba: {accuracy}')

# Imprimir el reporte de clasificación
print(f'Reporte de clasificación:\n{report}')

#segundo cambio

