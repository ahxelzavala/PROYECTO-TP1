# Proyecto de Análisis Predictivo con Machine Learning

Este proyecto implementa modelos de machine learning (Random Forest y XGBoost) para análisis predictivo de datos de ventas.

## Estructura del Proyecto

```
├── DATA/
│   ├── DATACASIFINAL.csv       # Dataset principal
│   └── DATAPERU2CSV.csv        # Dataset adicional
├── RANDOMFOREST.py             # Implementación de Random Forest
├── XGBOOST.py                 # Implementación de XGBoost
└── XGBOOST2.py               # Implementación alternativa de XGBoost
```

## Características

- Análisis predictivo usando Random Forest y XGBoost
- Preprocesamiento de datos con sklearn
- Evaluación de modelos con métricas de clasificación
- Manejo de datos categóricos y numéricos

## Requisitos

- Python 3.x
- pandas
- scikit-learn
- xgboost

## Uso

1. Asegúrese de tener instaladas todas las dependencias
2. Ejecute los scripts de Python según el modelo deseado:
   - `python RANDOMFOREST.py` para Random Forest
   - `python XGBOOST.py` o `python XGBOOST2.py` para XGBoost

## Modelos Implementados

### Random Forest
- Utiliza validación cruzada
- Incluye preprocesamiento de datos
- Genera reportes de clasificación

### XGBoost
- Implementa pipeline de procesamiento
- Maneja variables categóricas y numéricas
- Incluye evaluación de precisión