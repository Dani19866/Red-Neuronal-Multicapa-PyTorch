# 🤖 Clasificador Fashion MNIST con PyTorch y GUI

¡Bienvenido! 👋 Este proyecto es una aplicación de escritorio con una interfaz gráfica (GUI) que te permite crear, cargar, entrenar y evaluar modelos de redes neuronales (MLP) usando **PyTorch**.

El objetivo principal es clasificar imágenes del popular dataset **Fashion MNIST** 👕👖👟.

---

## ✨ Características Principales

Esta aplicación te permite gestionar todo el ciclo de vida de un modelo de clasificación:

* **🧠 Creación de Redes:** Permite crear dos tipos de arquitecturas:
    * **Rectangular (MLP Clásico):** Define el número de capas ocultas y neuronas por capa.
    * **Piramidal (MLP con interpolación):** Red neuronal multicapas con arquitectura piramidal.
* **💾 Cargar y Guardar:**
    * Carga un modelo `.pth` previamente entrenado para evaluarlo.
    * Guarda tu modelo recién entrenado en un archivo `.pth`.
* **📊 Entrenamiento Flexible:**
    * Selecciona tus propios archivos `.csv` de entrenamiento y prueba.
    * Define el **optimizador** (Adam, SGD o RMSProp), **tasa de aprendizaje**, **número de épocas** y **tamaño del lote** (batch size).
* **🔄 Doble Modo de Clasificación:**
    * **10 Clases:** Clasifica el dataset Fashion MNIST original (Camisa, Pantalón, Bota, etc.).
    * **4 Clases:** Agrupa las 10 clases en 4 categorías simplificadas (Top, Bottom, Calzado, Bolso).
* **📈 Evaluación del Modelo:**
    * Visualiza un **gráfico de pérdida (loss)** por época (Entrenamiento vs. Validación).
    * Muestra la **matriz de confusión** para analizar el rendimiento por clase.
    * Reporta métricas clave: **Accuracy, Precisión y Recall**.

---

## 🛠️ Tecnologías Utilizadas

Este proyecto fue construido con las siguientes herramientas y librerías:

* **Python 3**
* **PyTorch:** El framework principal para construir y entrenar las redes neuronales.
* **Tkinter:** Para la construcción de la interfaz gráfica de usuario (GUI).
* **Pandas:** Para la carga y manipulación inicial de los datos desde los archivos CSV.
* **Scikit-learn:** Para generar la matriz de confusión y las métricas de clasificación (accuracy, precision, recall).
* **Matplotlib & Seaborn:** Para la visualización de los gráficos de pérdida y la matriz de confusión.
* **Tqdm:** Para mostrar barras de progreso elegantes en la consola durante el entrenamiento.

---

## ⚠️ ¡Importante! Preparar los Datos

Este repositorio **no** incluye los archivos de datos. Sigue estos pasos para conseguirlos:

1.  Debes descargar el dataset **Fashion MNIST** en formato **CSV**.
2.  Puedes encontrarlo fácilmente en [Kaggle (Fashion MNIST)](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
3.  Necesitarás los archivos `fashion-mnist_train.csv` y `fashion-mnist_test.csv`.
4.  No necesitas ponerlos en una carpeta específica; la aplicación te pedirá que los selecciones usando un diálogo de archivos.

---

## 📈 Entendiendo las Métricas y Gráficos

La aplicación genera varias visualizaciones para ayudarte a entender el rendimiento de tu modelo. Aquí tienes una explicación detallada de qué significa cada una, extraída directamente de la documentación del código:

### 📊 Gráfico de Pérdida (Loss Curve)

* **DEFINICIÓN:** Este gráfico demuestra que tan bien está aprendiendo la MLP a lo largo del tiempo, es decir, a lo largo de las épocas.
* **Eje X:** Representa la cantidad de épocas.
* **Eje Y:** Medida de error. Un número alto hace referencia que el modelo está cometiendo muchos errores.
* **EXPLICACIÓN DE LÍNEAS:**
    * **Línea azul (Train Loss o pérdida de entrenamiento):** Mide el error del modelo sobre los datos que se están usando para aprender.
    * **Línea roja (Val Loss o pérdida de validación):** Mide el error del modelo sobre un conjunto de datos nuevos (test dataset). Representa la generalización del modelo.
* **OBJETIVO:** Lograr que las líneas converjan. Si la línea roja (medición del modelo vs nuevos datos) se aleja de la línea azul (medición del modelo vs datos de entrenamiento), entonces el modelo está memorizando, pero no está aprendiendo.

### 📉 Matriz de Confusión

* **DEFINICIÓN:** La matriz de confusión es una tabla que compara las etiquetas verdaderas (lo que realmente es) y las etiquetas predichas (lo que predijo el modelo).
* **EJE Y (etiqueta verdadera):** Cada fila representa la etiqueta real. Por ejemplo, la fila "0" contiene todas las imágenes que realmente eran de la clase "0".
* **EJE X (etiqueta predicha):** Cada columna representa la predicción del modelo. Por ejemplo, la columna "0" todo lo que el modelo predijo como "0".
* En conclusión, la matriz nos dice qué tan bueno es reconociendo ciertos objetos `[n, n]`, y qué tan malo es reconociendo otros `[n, m]`.
* **EXPLICACIÓN DE LA MATRIZ:** `[fila, columna]`
    * La celda `[1, 1] = 984`. El modelo predijo exitosamente 984 imágenes de clase "0".
    * La celda `[6, 0] = 142`. El modelo se equivocó haciendo la predicción, dijo que hay 142 imágenes de tipo "0", pero en verdad son de tipo "6".
* **OBJETIVO:** Tener los números más altos posibles en la diagonal principal. Tener los números más bajos posibles en el resto de la matriz.

### 🎯 Métricas de Precisión (Reporte de Clasificación)

* **DEFINICIÓN:** Esto muestra un reporte de clasificación, y da un resumen detallado, clase por clase, de qué tan bien funciona el modelo.
* **EXPLICACIÓN DE LAS MÉTRICAS:**
    * **Columna PRECISIÓN:** De todas las veces que el modelo predijo una clase, ¿Qué porcentaje de veces acertó?
        * Si tiene una precisión del 0.61 para la etiqueta "6", significa que de todas las imágenes que el modelo predijo de etiqueta "6", solo el 61% eran realmente de etiqueta "6". El resto de porcentaje quiere decir el porcentaje de fallos.
    * **Columna RECALL (Sensibilidad, tasa de verdaderos positivos):** La tasa de verdaderos positivos es la proporción de todos los positivos reales que se clasificaron correctamente como positivos.
        * Un modelo perfecto no tendría ningún falso negativo (muestras positivas que se clasificaron como negativo), por lo tanto, la tasa de verdaderos positivos sería 1.0, es decir, detecta el 100% de todas las etiquetas de ese tipo.
        * Un falso negativo tiene más consecuencias que un falso positivo.
    * **PRECISIÓN vs RECALL:**
        * RECALL calcula el porcentaje de etiquetas positivas que fueron identificadas.
        * PRECISIÓN: De todas las etiquetas identificadas, ¿Qué procentaje es realmente esa etiqueta?
    * **¿Qué priorizar?**
        * Priorizar **RECALL** cuando no importa la precisión. Esto tiene sentido cuando no se permite dejar pasar falsos positivos. Por ejemplo, en una detección de cáncer, no se permite dejar pasar un caso.
        * Priorizar **PRECISIÓN** cuando no importa el recall. Esto tiene sentido cuando se quiere tener menos falsos positivos. Por ejemplo, en el caso de un correo electrónico que le llega spam, se quiere tener mayor precisión, aunque esto signifique mayor probabilidad de correo spam en la bandeja de entrada.
    * **Columna F1-score:**
        * Es una métrica que combina la precisión y el recall en un solo número. Es la medida "armónica" que encuentra el mejor equilibrio posible between esas dos medidas.
        * Un F1-Score es alto solo si tanto la precisión como el recall son altos.
    * **Columna Support:**
        * El support es la cantidad de muestras totales que existen para cada clase en el conjunto de prueba. Esto solo es un conteo, no es una métrica.
    * **Medida de exactitud (Accuracy):**
        * Es la métrica más simple de todas. Del total de generalizaciones hechas, ¿Qué porcentaje fue correcto?
        * Fórmula: `(TOTAL DE ACIERTOS) / (TOTAL DE MUESTRAS)`.
        * Pero esta medida es engañosa, se debe combinar con el recall. Por ejemplo, un modelo que predice si una persona tiene cáncer y el 99% de los casos dice que no, entonces tiene una exactitud del 99%, pero es inútil porque el recall es del 0%.
