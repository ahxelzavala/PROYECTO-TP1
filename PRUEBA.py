import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar los datos
df = pd.read_csv('C:/Users/AhxelLuis/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PROYECTOTP/DATA/DATAPERU2CSV.csv')

# Filtrar el DataFrame por el tipo de venta
df = df.loc[df['TIPO_VENTA'] == 'Venta Stock']

# Preprocesar los datos eliminando columnas innecesarias
df.drop(columns = ['SOCIEDAD', 'N4', 'N41', 'Modulo', 'TIPO_CLIENTE.1', 'Lotes_y_cantidades', 'Moneda T/C', 'TIPO_VENTA'], inplace=True)

# Eliminar filas duplicadas
df.drop_duplicates(inplace=True)

# Eliminar filas con valores nulos (si existen)
df.dropna(inplace=True)

# Convertir las variables categóricas a valores numéricos si es necesario
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

# Codificar todas las columnas categóricas
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Codificar la columna 'SKU' (que es alfanumérica) a clases numéricas consecutivas
sku_encoder = LabelEncoder()
df['SKU'] = sku_encoder.fit_transform(df['SKU'])

# Verificar las clases de 'SKU' para asegurarnos de que son secuenciales
print(f"Clases de 'SKU': {sorted(df['SKU'].unique())}")

# Definir las características (X) y la variable objetivo (y)
X = df.drop(columns=['SKU'])  # Usamos todas las columnas excepto 'SKU'
y = df['SKU']  # La variable objetivo es 'SKU'

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características (opcional, pero puede mejorar el rendimiento si las variables tienen rangos muy diferentes)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y ajustar el modelo XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la precisión y el reporte de clasificación
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Mostrar los resultados
print(f"Precisión del modelo: {accuracy}")
print("Reporte de clasificación:\n", report)






