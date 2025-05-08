import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv('C:/Users/AhxelLuis/OneDrive - Universidad Peruana de Ciencias/Tesis_Serna_Zavala/PROYECTOTP/DATA/DATAPERU2CSV.csv')


print(df.head())

print(df.isnull().sum(axis=1).sort_values(ascending=False))# Filtrar el DataFrame por la columna TIPO_VENTA donde el valor sea "Venta Stock"
df = df.loc[df['TIPO_VENTA'] == 'Venta Stock']

# Preprocesar los datos
df.drop(columns = 'SOCIEDAD', inplace=True)
df.drop(columns = 'N4', inplace=True)    
df.drop(columns = 'N41', inplace=True)   
df.drop(columns = 'Modulo', inplace=True)    
df.drop(columns = 'TIPO_CLIENTE.1', inplace=True) 
df.drop(columns = 'Lotes_y_cantidades', inplace=True)
df.drop(columns = 'Moneda T/C', inplace=True)
df.drop(columns = 'TIPO_VENTA', inplace=True)

print(df.nunique().sort_values(ascending=True))

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print(df.head())



