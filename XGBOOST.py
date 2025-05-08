import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Cargar y filtrar datos
df = pd.read_csv('C:/Users/CARLOS SERNA/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PROYECTOTP/DATA/DATAPERU2CSV.csv')
df = df[df['TIPO_VENTA'] == 'Venta Stock'].head(1000)  # Limitar a 1000 registros para prueba

# Limpiar datos numéricos
for col in ['CANT', 'P_UNIT', 'V_TOTAL']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# Eliminar columnas no necesarias
columnas_a_eliminar = ['SOCIEDAD', 'N4', 'N41', 'Modulo', 'TIPO_CLIENTE.1', 'Lotes_y_cantidades', 'Moneda T/C', 'TIPO_VENTA']
df.drop(columns=columnas_a_eliminar, inplace=True)

# Limpiar datos
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Codificar variables categóricas
le = LabelEncoder()
df['SKU'] = le.fit_transform(df['SKU'])
df['TIPO_CLIENTE'] = LabelEncoder().fit_transform(df['TIPO_CLIENTE'])
df['ARTICULO'] = LabelEncoder().fit_transform(df['ARTICULO'])

# Preparar datos
X = df[['TIPO_CLIENTE', 'ARTICULO', 'CANT', 'P_UNIT', 'V_TOTAL']]
y = df['SKU']

# Normalizar variables numéricas
scaler = StandardScaler()
X[['CANT', 'P_UNIT', 'V_TOTAL']] = scaler.fit_transform(X[['CANT', 'P_UNIT', 'V_TOTAL']])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(df['SKU'].unique()),
    max_depth=3,
    learning_rate=0.1,
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nPrecisión del modelo: {accuracy:.2%}')

# Imprimir reporte detallado
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred))



