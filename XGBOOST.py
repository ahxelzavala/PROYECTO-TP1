import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv('C:/Users/AhxelLuis/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PRUEBAS2/DATA/Pruebasdatacasifinal.csv')

# Limpiar las columnas numéricas eliminando comas y convirtiéndolas a float
columns_to_clean = ['CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']
for col in columns_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True).astype(float)

# Crear una variable objetivo ficticia para productos complementarios
# Vamos a asumir que si el V_TOTAL > 1000, el producto es complementario
df['comprar_complementario'] = (df['V_TOTAL'] > 1000).astype(int)

# Seleccionar características y variable objetivo
features = ['TIPO_VENTA', 'TIPO_CLIENTE', 'SKU', 'ARTICULO', 'CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']
X = df[features]
y = df['comprar_complementario']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento de datos: Normalizar las características numéricas y codificar las categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['CANT', 'COSTO', 'C_TOTAL', 'P_UNIT', 'V_TOTAL', 'UTILIDAD']),  # Normalizar características numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['TIPO_VENTA', 'TIPO_CLIENTE', 'SKU', 'ARTICULO'])  # Codificación one-hot
    ])

# Crear el pipeline para entrenar el modelo XGBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42))  # XGBoost con 100 árboles y profundidad máxima de 10
])

# Entrenar el modelo
model_pipeline.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model_pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Imprimir los resultados
print(f'Precisión del modelo en los datos de prueba: {accuracy}')
print(f'Reporte de clasificación:\n{report}')
