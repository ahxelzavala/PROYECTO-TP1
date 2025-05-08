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

# Preprocesar los datos
df.drop(columns = 'SOCIEDAD', inplace=True)
df.drop(columns = 'N4', inplace=True)    
df.drop(columns = 'N41', inplace=True)   
df.drop(columns = 'Modulo', inplace=True)    
df.drop(columns = 'TIPO_CLIENTE.1', inplace=True)  

print(df.nunique().sort_values(ascending=True))

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print(df.duplicated().sum())

print(df.shape)