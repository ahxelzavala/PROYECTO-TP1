import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # Para sobre-muestreo de la clase minoritaria

# Cargar los datos
df = pd.read_csv('C:/Users/AhxelLuis/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PROYECTOTP/DATA/DATAPERU3CSV.csv')

# Filtrar el DataFrame por el tipo de venta (si es necesario)
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

# Crear la columna de "Compra Complementaria" como target (0 o 1)
# En este ejemplo, la lógica es asumir que si un cliente ha comprado más de un producto, se considera que compró un complemento
df['Compra_Complementaria'] = (df.duplicated(subset=['CODIGO', 'RAZON_SOCIAL'], keep=False)).astype(int)

# Verificar las clases de 'Compra_Complementaria'
print(f"Clases de 'Compra_Complementaria': {sorted(df['Compra_Complementaria'].unique())}")

# Definir las características (X) y la variable objetivo (y)
X = df.drop(columns=['Compra_Complementaria', 'SKU'])  # Usamos todas las columnas excepto 'Compra_Complementaria' y 'SKU'
y = df['Compra_Complementaria']  # La variable objetivo es 'Compra_Complementaria'

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características (opcional, pero puede mejorar el rendimiento si las variables tienen rangos muy diferentes)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Manejo de clases desbalanceadas**: Sobremuestreo usando SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Crear y ajustar el modelo XGBoost con ajuste de peso (opcional si usas SMOTE)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
model.fit(X_train_resampled, y_train_resampled)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la precisión y el reporte de clasificación
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Evaluar el modelo con AUC-ROC para tener una mejor métrica para el desbalance de clases
auc_score = roc_auc_score(y_test, y_pred)

# Mostrar los resultados
print(f"Precisión del modelo: {accuracy}")
print("Reporte de clasificación:\n", report)
print(f"AUC-ROC: {auc_score}")

