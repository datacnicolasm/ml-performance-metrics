# Métricas de Evaluación de Modelos de Machine Learning

Este repositorio contiene ejemplos prácticos en Python sobre cómo calcular diversas métricas de evaluación para modelos de **Machine Learning**.

## ¿Qué son las métricas de evaluación de modelos?

Las métricas de evaluación son herramientas esenciales para medir el rendimiento de un modelo de Machine Learning. Nos permiten entender qué tan bien se está desempeñando el modelo, ya sea en tareas de clasificación o regresión. En modelos de clasificación, estas métricas son fundamentales para determinar qué tan acertadas son las predicciones del modelo en diferentes escenarios.

## 1. Precisión en un Modelo de Clasificación

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

### Explicación:

1. **Cargar datos:** Aquí utilizamos el conjunto de datos Iris (puedes reemplazarlo por tus propios datos).
2. **División de datos:** Separamos los datos en un conjunto de entrenamiento (70%) y uno de prueba (30%).
3. **Entrenamiento:** Entrenamos un modelo de clasificación con Random Forest.
4. **Precisión:** Utilizamos la función `precision_score` de `scikit-learn` para calcular la precisión del modelo en los datos de prueba.

En el caso de multiclasificación, utilizamos `average='macro'` para calcular la precisión promedio por clase. Si el problema es binario, puedes omitir el argumento `average`.

### ¿Cómo se interpreta el valor de la precisión?

- **Precisión** = 1.0 (100%): El modelo tiene una precisión perfecta, lo que significa que cada vez que predijo una clase positiva, acertó. Esto indica que el modelo no cometió errores al clasificar ejemplos negativos como positivos.
- **Precisión** < 1.0: Si la precisión fuera, por ejemplo, 0.8 (80%), significaría que el 80% de las veces que el modelo predijo una clase positiva, acertó, pero el 20% de las predicciones positivas fueron en realidad falsas (falsos positivos).

Un valor de precisión perfecto (1.0) puede ser indicativo de que el modelo está funcionando muy bien en este conjunto de datos, o que los datos están altamente equilibrados o son fáciles de clasificar.

Sin embargo, la precisión por sí sola no siempre es suficiente para evaluar el rendimiento global del modelo, ya que ignora el impacto de los **falsos negativos**. Es importante combinar la **precisión** con otras métricas como el **recall** (sensibilidad) y el **F1-Score** para obtener una imagen completa del rendimiento.

## 2. Recall (Sensibilidad)

El recall, también conocido como sensibilidad o tasa de verdaderos positivos, mide la capacidad del modelo para identificar correctamente las instancias positivas de una clase. Es especialmente útil cuando es importante capturar todos los casos positivos, incluso si eso significa obtener algunos falsos positivos.

En Python, utilizando librerías como **scikit-learn**, puedes calcular de forma sencilla:

```python
# Importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
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

# Calcular el Recall
recall = recall_score(y_test, y_pred, average='macro')

# Mostrar el resultado
print(f'Recall del modelo: {recall:.2f}')
```
### Explicación:

1. **Cargar el conjunto de datos:** Utilizamos el conjunto de datos Iris para este ejemplo, que es un dataset de clasificación multiclase.
2. **División de datos:** Separamos los datos en conjuntos de entrenamiento y prueba (70% para entrenamiento y 30% para prueba).
3. **Entrenamiento del modelo:** Entrenamos un modelo de clasificación con Random Forest.
4. **Predicción:** Usamos el modelo entrenado para hacer predicciones en los datos de prueba.
5. **Cálculo del Recall:** Utilizamos la función `recall_score` de `scikit-learn`, con el parámetro `average='macro'` para calcular el recall medio en un escenario de clasificación multiclase.
6. **Resultado:** El valor del `recall` se imprime en pantalla.

### ¿Cómo se interpreta el valor de recall?

El recall mide cuántos de los verdaderos positivos fueron detectados correctamente por el modelo.

- Un valor de recall = 1.0 indica que el modelo detectó correctamente todos los verdaderos positivos, sin dejar escapar ninguno.
- Un recall bajo indica que el modelo está fallando en capturar una parte significativa de los verdaderos positivos, lo que podría ser problemático en aplicaciones donde es más importante no perder ningún positivo (como en la detección de enfermedades o fraudes).

La precisión y el recall están relacionados pero se enfocan en diferentes aspectos del rendimiento del modelo:

- Precisión mide qué tan bien el modelo evita los falsos positivos. Es ideal cuando el costo de un falso positivo es alto (por ejemplo, en una predicción de spam).
- Recall mide qué tan bien el modelo detecta los verdaderos positivos. Es útil cuando es más importante capturar todos los casos positivos, incluso a costa de cometer más falsos positivos.

Un modelo puede tener:

- Alta precisión y bajo recall: El modelo es bueno para evitar falsos positivos, pero puede estar perdiendo muchos verdaderos positivos.
- Alto recall y baja precisión: El modelo está detectando la mayoría de los positivos, pero también genera muchos falsos positivos.
