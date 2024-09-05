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

- Un valor de **recall = 1.0** indica que el modelo detectó correctamente todos los verdaderos positivos, sin dejar escapar ninguno.
- Un recall bajo indica que el modelo está fallando en capturar una parte significativa de los verdaderos positivos, lo que podría ser problemático en aplicaciones donde es más importante no perder ningún positivo (como en la detección de enfermedades o fraudes).

La precisión y el recall están relacionados pero se enfocan en diferentes aspectos del rendimiento del modelo:

- Precisión mide qué tan bien el modelo evita los falsos positivos. Es ideal cuando el costo de un falso positivo es alto (por ejemplo, en una predicción de spam).
- Recall mide qué tan bien el modelo detecta los verdaderos positivos. Es útil cuando es más importante capturar todos los casos positivos, incluso a costa de cometer más falsos positivos.

Un modelo puede tener:

- **Alta precisión y bajo recall:** El modelo es bueno para evitar falsos positivos, pero puede estar perdiendo muchos verdaderos positivos.
- **Alto recall y baja precisión:** El modelo está detectando la mayoría de los positivos, pero también genera muchos falsos positivos.

## 3. Exactitud (Accuracy)

La exactitud o accuracy es una métrica que mide qué tan bien un modelo clasifica correctamente tanto las clases positivas como las negativas. Es la proporción de predicciones correctas sobre el total de predicciones realizadas. En otras palabras, mide el porcentaje de predicciones correctas que hizo el modelo sobre todos los casos.

La exactitud es útil cuando las clases están equilibradas. Sin embargo, en casos de clases desbalanceadas (por ejemplo, una clase positiva muy poco frecuente), puede ser engañosa, ya que un modelo que siempre predice la clase mayoritaria tendrá una alta exactitud, pero no detectará la clase minoritaria.

En Python, utilizando librerías como **scikit-learn**, puedes calcular de forma sencilla:

```python
# Importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar un modelo de bosque aleatorio
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predecir los valores del conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la exactitud (accuracy)
accuracy = accuracy_score(y_test, y_pred)

# Mostrar el resultado
print(f'Exactitud del modelo: {accuracy:.2f}')
```

### Explicación:

1. **Cargar el conjunto de datos:** Utilizamos el conjunto de datos Iris para este ejemplo, que es un dataset de clasificación multiclase.
2. **División de datos:** Separamos los datos en conjuntos de entrenamiento y prueba (70% para entrenamiento y 30% para prueba).
3. **Entrenamiento del modelo:** Entrenamos un modelo de clasificación con Random Forest.
4. **Predicción:** Usamos el modelo entrenado para hacer predicciones en los datos de prueba.
5. **Cálculo del Recall:** Utilizamos la función `accuracy_score(y_test, y_pred)` que calcula la exactitud comparando las etiquetas reales `(y_test)` con las predicciones realizadas `(y_pred)`.
6. **Resultado:** El resultado es un valor numérico que indica el porcentaje de predicciones correctas sobre el total de predicciones.

### ¿Cómo se interpreta el valor de Accuracy?

La exactitud (o accuracy) indica qué porcentaje de predicciones totales fueron correctas. En el ejemplo anterior, si el resultado fuera `0.93`, significaría que el modelo predijo correctamente el 93% de los ejemplos en el conjunto de prueba.

**Interpretación:**

- **Exactitud alta (cercana a 1):** El modelo está clasificando correctamente la mayoría de los casos. Una exactitud alta es ideal, especialmente en problemas donde las clases están equilibradas.
- **Exactitud baja (cercana a 0):** El modelo está clasificando mal una gran cantidad de casos. Puede necesitar ajustes en el modelo, más datos de entrenamiento, o tal vez las clases están desbalanceadas.

**Limitaciones:**

- **Datos desbalanceados:** Si una clase es mucho más frecuente que otra, el modelo puede obtener una alta exactitud simplemente prediciendo siempre la clase mayoritaria, sin aprender realmente a detectar la clase menos frecuente. En estos casos, métricas como precisión, recall o el F1-score pueden ser más útiles que la exactitud sola.

## 4. Curva ROC y AUC

La **Curva ROC (Receiver Operating Characteristic)** es una representación gráfica que muestra el rendimiento de un modelo de clasificación en todos los umbrales de predicción posibles. Se utiliza principalmente para problemas de clasificación binaria y evalúa qué tan bien el modelo puede distinguir entre dos clases (positiva y negativa).

La **AUC (Area Under the Curve)** es el área bajo la curva ROC. Es una métrica que resume el rendimiento del modelo en un solo valor, que varía entre 0 y 1:

- **AUC = 1:** El modelo es perfecto en su clasificación.
- **AUC = 0.5:** El modelo es tan bueno como hacer predicciones al azar.
- **AUC < 0.5:** El modelo clasifica peor que el azar, lo que indica que algo está mal con el modelo.

En Python, utilizando librerías como **scikit-learn** y **matplotlib** lo puedes calcular de forma sencilla:

```python
# Importar librerías necesarias
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Cargar conjunto de datos
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar un modelo de bosque aleatorio
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Obtener las probabilidades de predicción
y_prob = model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calcular el AUC
auc = roc_auc_score(y_test, y_prob)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Mostrar el valor del AUC
print(f'AUC del modelo: {auc:.2f}')
```

### Explicación:

1. Cargar datos: Usamos el conjunto de datos de cáncer de mama de `scikit-learn` para este ejemplo, que es un conjunto de datos binario (clases positivas y negativas).
2. Dividir datos: Dividimos el conjunto de datos en entrenamiento y prueba (70% entrenamiento y 30% prueba) para evaluar el rendimiento del modelo.
3. Entrenar modelo: Usamos un Random Forest como modelo de clasificación. Puedes cambiar el modelo si lo deseas.
4. Obtener probabilidades: `predict_proba` devuelve las probabilidades de pertenencia a cada clase. Aquí seleccionamos la probabilidad de que una observación pertenezca a la clase positiva `([:, 1])`.
5. Calcular la curva ROC: La función `roc_curve` calcula los valores de la tasa de falsos positivos (FPR) y la tasa de verdaderos positivos (TPR) para varios umbrales.
6. Calcular el AUC: `roc_auc_score` calcula el área bajo la curva (AUC), un resumen del rendimiento del modelo.
7. Graficar la curva ROC: Usamos `matplotlib` para trazar la curva ROC. La diagonal roja es la línea de referencia, que representa un modelo que predice al azar `(AUC = 0.5)`.

### ¿Cómo se interpreta?

**Curva ROC:**

- La curva ROC representa el rendimiento del modelo a lo largo de varios umbrales de clasificación. El eje Y (TPR) muestra la sensibilidad o tasa de verdaderos positivos, y el eje X (FPR) muestra la tasa de falsos positivos.
- Una curva que se acerca al punto superior izquierdo (donde TPR = 1 y FPR = 0) indica un buen rendimiento, ya que significa que el modelo tiene una alta tasa de verdaderos positivos y una baja tasa de falsos positivos.

**AUC (Área bajo la curva ROC):**

- El valor de AUC es una métrica que resume la capacidad del modelo para distinguir entre las clases.
- **AUC = 1.0:** El modelo es perfecto y separa completamente las clases positivas y negativas.
- **AUC = 0.5:** El modelo no tiene capacidad de discriminación, es decir, su rendimiento es equivalente a hacer predicciones al azar.
- **AUC < 0.5:** Indica que el modelo clasifica peor que al azar, lo que sugiere que está haciendo algo incorrecto, como invertir las clases.

## 5. F1-Score

El F1-Score es la métrica que combina la precisión y el recall (sensibilidad) en un solo valor. Es especialmente útil cuando las clases están desbalanceadas o cuando tanto los falsos positivos como los falsos negativos son importantes.

En Python, utilizando librerías como **scikit-learn**, puedes calcular de forma sencilla:

```python
# Importar las librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
data = load_iris()
X = data.data
y = data.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el F1-Score
f1 = f1_score(y_test, y_pred, average='macro')

# Mostrar el F1-Score
print(f'F1-Score del modelo: {f1:.2f}')
```

### Explicación:

1. **Importación de librerías:** Importamos las funciones necesarias de `scikit-learn`, incluyendo el clasificador de bosque aleatorio (`RandomForestClassifier`), la función para calcular el F1-Score (`f1_score`), y el conjunto de datos Iris (`load_iris`).
2. **Cargar el conjunto de datos:** Utilizamos el conjunto de datos Iris como ejemplo, donde X contiene las características (variables predictoras) y y contiene las etiquetas de las clases.
3. **División del conjunto de datos:** Dividimos los datos en entrenamiento (70%) y prueba (30%) utilizando train_test_split.
4. **Entrenamiento del modelo:** Entrenamos un clasificador de bosque aleatorio con el conjunto de datos de entrenamiento.
5. **Predicción:** Utilizamos el modelo para predecir las etiquetas de las muestras del conjunto de prueba.
6. **Cálculo del F1-Score:** Utilizamos la función `f1_score` de `scikit-learn` para calcular el F1-Score. El parámetro `average='macro'` se utiliza para calcular el F1-Score promedio de todas las clases en un problema de clasificación multiclase.
7. **Mostrar el resultado:** Finalmente, se imprime el F1-Score obtenido para el modelo.

### ¿Cómo se interpreta?

El F1-Score combina la precisión y el recall en una sola métrica, siendo particularmente útil cuando hay un equilibrio entre la importancia de ambos.

- **Si el F1-Score es cercano a 1:** El modelo tiene un buen equilibrio entre precisión y recall. En este caso, tanto los falsos positivos como los falsos negativos son bajos, lo que indica que el modelo está prediciendo bien tanto las clases positivas como las negativas.
- **Si el F1-Score es bajo (cercano a 0):** Esto indica que el modelo tiene un mal rendimiento, ya sea en precisión, recall o ambos. Un valor bajo puede significar que el modelo está cometiendo muchos errores en la predicción de clases positivas o negativas.

El F1-Score es útil cuando:

- **Las clases están desbalanceadas.** Por ejemplo, si tienes un conjunto de datos donde hay muchas más muestras de una clase que de otra, el F1-Score te ayuda a evaluar el modelo de manera más justa que la precisión o la exactitud por sí solas.
- **La importancia de falsos positivos y falsos negativos es similar.** Si el costo de ambos errores es relevante, el F1-Score proporciona una buena medida de la calidad general del modelo.
