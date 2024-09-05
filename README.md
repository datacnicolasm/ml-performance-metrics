# Métricas de Evaluación de Modelos de Machine Learning

Este repositorio contiene ejemplos prácticos en Python sobre cómo calcular diversas métricas de evaluación para modelos de **Machine Learning**.

## ¿Qué son las métricas de evaluación de modelos?

Las métricas de evaluación son herramientas esenciales para medir el rendimiento de un modelo de Machine Learning. Nos permiten entender qué tan bien se está desempeñando el modelo, ya sea en tareas de clasificación o regresión. En modelos de clasificación, estas métricas son fundamentales para determinar qué tan acertadas son las predicciones del modelo en diferentes escenarios.

### Precisión en un Modelo de Clasificación

La precisión de un modelo de clasificación se calcula como la proporción de verdaderos positivos (TP) sobre el total de predicciones positivas (TP + FP). Es una métrica clave cuando el coste de los falsos positivos es alto.

En Python, utilizando librerías como **scikit-learn**, puedes calcular la precisión de forma sencilla:

```python
# Importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.datasets import load_iris

# Cargar conjunto de datos
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar un modelo de bosque aleatorio
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predecir los valores de prueba
y_pred = model.predict(X_test)

# Calcular la precisión
precision = precision_score(y_test, y_pred, average='macro')

# Mostrar el resultado
print(f'Precisión del modelo: {precision:.2f}')
```
