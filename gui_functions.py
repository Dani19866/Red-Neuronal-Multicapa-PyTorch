from mlp import MLP
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from utils import seleccionar_archivo

# LISTO: Creamos el modelo (rectangular o convolucional)
def create_model(tipo_de_red:str, neuronas_entrada: int, neuronas_salida: int, capas_ocultas:int, neuronas_por_capa: int = 0, agrupar_articulos: bool = False):
    # tipo_de_red = rectangular | convolucional
    mlp = MLP(int(neuronas_entrada), int(neuronas_salida), int(capas_ocultas), int(neuronas_por_capa))
    
    # Verificar si es rectangular o convolucional
    mlp.create_rectangular_network() if tipo_de_red == "rectangular" else mlp.create_convolucional_network()
    
    # Asignar si es Fashion MNIST o AGRUPAR ARTÍCULOS
    mlp.groupFashionMNIST = True if agrupar_articulos else False
    
    return mlp

# LISTO: Entrenamos el modelo según la clasificación: mnist o grupo de artículos
def train_model(mlp: MLP, tipo_de_clasificacion: str, optimizador:str, tasa_aprendizaje: float, epocas:int, tamaño_de_lote: int, train_dataset_path, test_dataset_path):
    # Preparación de archivos de entrenamiento
    # Verificar si es mnist o articulos
    train_dataset = None
    test_dataset = None
    
    # Sirve para modelos ya creados
    print("\n-------------------------------------------------------------")
    print(f"DEBUG: ¿El modelo clasifica por grupo de artículos? {mlp.groupFashionMNIST}")
    print("-------------------------------------------------------------\n")
    
    # Tipo de dataset: MNIST
    if not mlp.groupFashionMNIST:
        train_dataset = dataset_train_loader(tamaño_de_lote=int(tamaño_de_lote), filepath=train_dataset_path)
        test_dataset = dataset_test_loader(filepath=test_dataset_path, tamaño_de_lote=int(tamaño_de_lote))
    
    # Tipo de dataset: GRUPO DE ARTÍCULOS
    else:
        train_dataset = dataset_train_loader(group=True, tamaño_de_lote=int(tamaño_de_lote), filepath=train_dataset_path)
        test_dataset = dataset_test_loader(group=True, filepath=test_dataset_path, tamaño_de_lote=int(tamaño_de_lote))
        mlp.groupFashionMNIST = True
        
    if train_dataset is None or test_dataset is None:
        messagebox.showerror("Error de Datos", "No se pudieron cargar los archivos de dataset. Revise las rutas o los archivos.")
        return # Detiene el entrenamiento
    
    # Funcionamiento correcto, entonces se setean las listas
    mlp.set_train_dataset(train_dataset)
    mlp.set_test_dataset(test_dataset)
    
    # Comenzar entrenamiento
    continue_trainig = mlp.trained
    tasa_aprendizaje = float(tasa_aprendizaje)
    epocas = int(epocas)
    
    # print("\n-------------------------------------------------------------")
    # print(f"DEBUG: ¿El modelo ha sido entrenado? {continue_trainig}")
    # print("-------------------------------------------------------------\n")
    
    
    # Primera vez entrenando
    if not continue_trainig:
        # print("\n-------------------------------------------------------------")
        # print(f"DEBUG: El modelo va a ser entrenado POR PRIMERA VEZ")
        # print("-------------------------------------------------------------\n")
        mlp.train_model(
            optimizador,
            tasa_aprendizaje,
            epocas
        )
    # Continuar entrenando, se salvan los datos anteriores
    else:
        # print("\n-------------------------------------------------------------")
        # print(f"DEBUG: El modelo va a ser entrenado y conservará los datos anteriores")
        # print("-------------------------------------------------------------\n")
        mlp.train_model(
            optimizador,
            tasa_aprendizaje,
            epocas,
            continue_trainig
        )

# LISTO: Guarda el modelo en .pth
def save_model(mlp: MLP, tipo_de_red: str):
    mlp.save_model(tipo_de_red)
    
# LISTO: Carga el modelo con todos sus hiperparámetros
def load_model(model_path):
    return MLP.load_model_from_file(model_path)

# LISTO: Evalua el modelo con el dataset de prueba
def test_model(mlp: MLP, new_path = False, tamaño_de_lote = 64):
    # Si se usa en las métricas
    if not new_path:
        print("\n-------------------------------------------------------------")
        print("DEBUG: Entrando en el test del modelo SIN ELEGIR NUEVO ARCHIVO")
        print("-------------------------------------------------------------\n")
        
        # Llamar al método de la clase MLP que hace el trabajo
        try:
            # Aquí se llama al método de la CLASE MLP
            y_pred, y_true = mlp.test_model() 
            
            if y_pred is None or y_true is None:
                # Esto puede pasar si el método interno falló
                messagebox.showerror("Error", "La evaluación del modelo falló internamente. Revise la consola.")
                return None, None
            
            # Si todo salió bien, devuelve los resultados
            return y_pred, y_true
            
        except Exception as e:
            messagebox.showerror("Error de Evaluación", f"Ocurrió un error inesperado durante la evaluación: {e}")
            print(f"ERROR en gui_functions.test_model: {e}")
            return None, None
        
    # Si se usa con otro path
    else:
        print("\n-------------------------------------------------------------")
        print("DEBUG: Entrando en el test del modelo ELIGIENDO NUEVO ARCHIVO")
        print("-------------------------------------------------------------\n")
        
        # Selección del archivo nuevo de prueba
        filepath = seleccionar_archivo(
            "Seleciona el nuevo archivo a probar",
            [
                ("Archivos CSV", "*.csv"),
                ("Todos los archivos", "*")
            ]
        )
        
        # Verificar si se eligió una ruta
        if filepath:
            # Mensaje en pantalla
            messagebox.showinfo("Información",f"Ruta de archivo: {filepath}\nTamaño de lote: {tamaño_de_lote}\nConjunto de grupo: {mlp.groupFashionMNIST}")
            messagebox.showinfo("Información", "El archivo se usará como nuevo dataset de test en la red neuronal")

            # Preparar dataset
            new_dataset_test = None
            
            # Si la configuración guardada del MLP es de grupo
            if not mlp.groupFashionMNIST:
                new_dataset_test = dataset_test_loader(tamaño_de_lote=int(tamaño_de_lote), filepath=filepath)
            
            # Si es Fashion MNIST
            else:
                new_dataset_test = dataset_test_loader(group=mlp.groupFashionMNIST, tamaño_de_lote=int(tamaño_de_lote), filepath=filepath)

            # Setear el conjunto de datos
            mlp.set_test_dataset(new_dataset_test)
            
            # Obtener datos de test y mostrar estadísticas
            precision_metrics(mlp)
            confusion_matrix(mlp)
        
        else:
            messagebox.showerror("Error en la selección de archivo", "No se seleccionó ningún archivo")

# LISTO: Muestra un gráfico de linea: compara pérdida de entrenamietno y validación
def loss_graphic(mlp: MLP):
    """
    Curva de périda (Loss Curve)
    
    DEFINICIÓN
        Este gráfico demuestra que tan bien está aprendiendo la MLP a lo 
        largo del tiempo, es decir, a lo largo de las épocas
        
        Eje X: Representa la cantidad de épocas
        Eje Y: Medida de error. Un número alto hace referencia que el modelo
        está comentiendo muchos errores.
        
    EXPLICACIÓN DE LINEAS:
        Línea azul (Train Loss o pérdida de entrenamiento): Mide el error
        del modelo sobre los datos que se están usando para aprender.
        
        Linea roja (Val Loss o pérdida de validación): Mide el error del
        modelo sobre un conjunto de datos nuevos (test dataset). Representa
        la generalización del modelo.
        
    OBJETIVO
        Lograr que las lineas convergan
        
        Si la línea roja (medición del modelo vs nuevos datos) se aleja
        de la línea azul (medición del modelo vs datos de entrenamiento),
        entonces el modelo está memorizando, pero no está aprendiendo.
    
    """
    
    # Verificar si hay historial
    if not mlp.trained:
        messagebox.showwarning("Sin datos", "No hay historial de pérdidas. Debe entrenar el modelo primero.")
        return
    
    # Crear el gráfico
    epochs = range(1, len(mlp.train_losses) + 1)
    
    # Crea una nueva figura
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mlp.train_losses, 'b-o', label='Pérdida de Entrenamiento (Train Loss)')
    plt.plot(epochs, mlp.val_losses, 'r-s', label='Pérdida de Validación (Val Loss)')
    
    # Añadir títulos y etiquetas
    plt.title('Historial de Pérdida (Loss) por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (Loss)')
    plt.legend()
    plt.grid(True)
    
    # 4. Mostrar el gráfico
    plt.tight_layout()
    plt.show()

# LISTO: Muestra la matriz de confusión usando el dataset de prueba
def confusion_matrix(mlp: MLP):
    """
    Matriz de confusión
    
    DEFINICIÓN
        La matriz de confusiín es una tabla que compara las etiquetas
        verdaderas (lo que realmente es) y las etiquetas predichas (lo que
        predijo el modelo).
        
        EJE Y (etiqueta verdadera): Cada fila representa la etiqueta real.
        Por ejemplo, la fila "0" contiene todas las imágenes que realmente
        eran de la clase "0".
        EJE X (etiqueta predicha): Cada columna representa la predicción
        del modelo. Por ejemplo, la columna "0" todo lo que el modelo
        predijo como "0"
        
        En conclusión, la matriz nos dice que tan bueno es reconociendo
        ciertos objetos [n, n], y que tan malo es reconociendo otros [n, m].
    
    EXPLICACIÓN DE LA MATRIZ
        [fila, columna]
    
        La celda [1, 1] = 984. El modelo predijo exitosamente 984 
        imágenes de clase "0"
        
        La celda [6, 0] = 142. El modelo se equivocó haciendo la predicción,
        dijo que hay 142 imágenes de tipo "0", pero en verdad son de tipo
        "6"
    
    OBJETIVO
        Tener los números más altos posibles en la diagonal principal.
        Tener los números mas bajos posibles en el resto de la matriz
    
    """
    
    # Verificar si ha sido entrenada la red
    if not mlp.trained:
        messagebox.showwarning("Sin entreamiento", "Primero deberá entrenar la red antes de verificar la matriz de confusión.")
        return
    
    # Comprobar si hay datasets
    if mlp.test_dataset is None:
        messagebox.showerror("Error de Datos", "No se ha hecho un test o no se ha entrenado la red, por lo tanto no hay datos para mostrar.")
        return
    
    # Obtener predicciones
    y_pred, y_true = test_model(mlp)
    
    # Verificación de error
    if y_pred is None or y_true is None:
        print("No se pudo generar la matriz de confusión (falló test_model).")
        return # test_model ya mostró un error
    
    # 2. Calcular la matriz
    cm = sk_confusion_matrix(y_true, y_pred)
    
    # Determinar las etiquetas (labels)
    # Usamos las neuronas de salida para saber cuántas clases hay (ej: 10 para mnist, 4 para group)
    num_classes = mlp.neuronas_salida
    class_labels = [str(i) for i in range(num_classes)]
    
    # 4. Graficar con Seaborn
    plt.figure(figsize=(10, 8)) # Nueva figura
    sns.heatmap(
        cm, 
        annot=True,     # Mostrar los números dentro de las celdas
        fmt='d',        # Formato de los números (entero)
        cmap='Blues',   # Paleta de colores
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera (True Label)')
    plt.xlabel('Etiqueta Predicha (Predicted Label)')
    plt.show()

# LISTO: Calcula las métricas de precisión, recall y accuracy
def precision_metrics(mlp: MLP):
    """
    Métricas de precisión
    
    DEFINICIÓN
        Esto muestra un reporte de clasificación, y da un resumen detallado,
        clase por clase, de que tan bien funciona el modelo
    
    EXPLICACIÓN DE LAS MÉTRICAS
        Columna PRECISIÓN:
            De todas las veces que el modelo predijo una clase, ¿Qué
            porcetaje de veces acertó?
            
            Si tiene una precisión del 0.61 para la etiqueta "6", significa
            que de todas las imágenes que el modelo predijo de etiqueta "6",
            solo el 61% eran realmente de etiqueta "6". El resto de porcentaje
            quiere decir el porcentaje de fallos.
            
        Columna RECALL (Sensibilidad, tasa de verdaderos positivos o probabilidad de detección)
            La tasa de verdaderos positivos es la proporción de todos los
            positivos reales que se clasificaron correctamente como
            positivos.

            Un modelo perfecto no tendría ningún falso negativo (muestras
            positivas que se clasificaron como negativo), por lo tanto,
            la tasa de verdaderos positivos sería 1.0, es decir, detecta
            el 100% de todas las etiquetas de ese tipo.
            
            Un falso negativo tiene más consecuencias que un falso positivo.
            
        PRECISIÓN vs TASA DE VERDADEROS POSITIVOS
            TASA DE VERDADEROS POSITIVOS calcula el porcentaje de etiquetas
            positivas fueron identificadas
            
            PRECISIÓN: De todas las etiquetas identificadas, ¿Qué procentaje
            es realmente esa etiqueta?
            
        ¿Qué priorizar entre RECALL y PRECISIÓN?
            Priorizar RECALL cuando no importa la precisión. Esto tiene sentido
            cuando no se permite dejar pasar falsos positivos. Por ejemplo,
            en una detección de cáncer, no se permite dejar pasar un caso. Eso
            significa que va a detectar falsos negativos; en el ejemplo del
            cáncer puede haber clientes que se hayan detectado con cáncer, pero
            realmente no lo tienen.
            
            Priorizar PRECISIÓN cuando no importa el recall. Esto tiene
            sentido cuando se quiere tener menos falsos positivos. Por ejemplo,
            en el caso de un correo electrónico que le llega spam, se quiere
            tener mayor precisión, aunque esto signifique mayor probabilidad
            de correo spam en la bandeja de entrada.
            
            La PRECISIÓN mejora a medida que disminuye los falsos positivos,
            mientras que el RECALL mejora cuando disminuyen los falsos
            negativos
            
            Si aumentas el Recall, tiendes a bajar la Precisión. (Atrapas 
            todos los peces, pero también mucha basura).
            
            Si aumentas la Precisión, tiendes a bajar el Recall. (Solo atrapas
            peces buenos, pero se te escapan muchos).
            
        Columna F1-score
            Es una métrica que combina la precisión y el recall en un solo
            número. Es la medida "armónica" que encuentra el mejor equilibrio
            posible entre esas dos medidas
            
            El F1-score expone una sola cifra que representa que tan bien
            generaliza el modelo logrando ambas cosas a la vez.
            
            Un F1-Score es alto solo si tanto la precisión como el recall 
            son altos. Un F1-Score cercano a 1: el modelo tiene mucha 
            precisión y mucho recall. Un F1-Score cercano a 0: el modelo
            no es preciso y poco recall.
            
        Columa Support
            El support es la cantidad de muestras totales que existen para
            cada clase en el conjunto de prueba. Esto solo es un conteo, no
            es una métrica. En el ejemplo, esto representa que hay 1000
            imágenes de etiqueta "1", "2", etc. 

        Medida de exactitud (Accuracy)
            Es la métrica más simple de todas. Del total de generalizaciones
            hechas, ¿Qué porcentaje fue correcto?
            
            Fórmula: (TOTAL DE ACIERTOS) / (TOTAL DE MUESTRAS)
            
            Pero esta medida es engañosa, se debe combinar con el recall
            para obtener más datos. Por ejemplo, un modelo que predice si
            una persona tiene cáncer y el 99% de los casos dice que no,
            entonces tiene una exactitud del 99%, pero es inútil porque
            el recall es del 0%. También el F1-Score te lo va a decir, porque
            tendría un valor de 0.0 a pesar de su 99% de exactitud
    """
    
    # Verificar si ha sido entrenada la red
    if not mlp.trained:
        messagebox.showwarning("Sin entreamiento", "Primero deberá entrenar la red antes de verificar la matriz de confusión.")
        return
    
    # Comprobar si hay datasets
    if mlp.test_dataset is None:
        messagebox.showerror("Error de Datos", "No se ha hecho un test o no se ha entrenado la red, por lo tanto no hay datos para mostrar.")
        return
    
    # Obtener predicciones (Llama a la función puente de la GUI)
    y_pred, y_true = test_model(mlp)
    
    # Verificaciones
    if y_pred is None or y_true is None:
        print("No se pudieron calcular las métricas (falló test_model).")
        return

    # Determinar el número de clases para el 'average'
    num_classes = mlp.neuronas_salida
    
    # Si es multiclase (más de 2), usamos un promedio 'macro' o 'weighted'
    avg_method = 'macro' if num_classes > 2 else 'binary'
    
    # Calcular métricas simples (zero_division=0 evita warnings si una clase nunca fue predicha)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
    
    # Obtener reporte detallado: Esto da precision/recall/f1-score por CADA clase
    report = classification_report(y_true, y_pred, zero_division=0)
    
    print("--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
    print(report)
    print("------------------------------------------")
    
    # Mostrar resumen en un messagebox
    summary_message = (
        f"Métricas (Promedio '{avg_method}'):\n"
        f"----------------------------------\n"
        f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"Precisión: {precision:.4f}\n"
        f"Recall (Sensibilidad): {recall:.4f}\n"
        f"----------------------------------\n"
        f"\nSe imprimió un reporte detallado por clase en la consola."
    )

    messagebox.showinfo("Métricas de Evaluación", summary_message)

# LISTO: Carga los datos de entrenamiento en un DataLoader
def dataset_train_loader(group = False, tamaño_de_lote: int = 64, filepath: str = None):
    if filepath:
        print(f"Cargando archivos de entrenamiento de la ruta: {filepath}")
    
        try:
            # Leer el archivo CSV con Pandas
            data_df = pd.read_csv(filepath)
            
            # 3. Separar etiquetas (y) y características (X)
            # Se supone que la primera columna es 'label' y el resto son píxeles
            labels_df = data_df.iloc[:, 0]
            features_df = data_df.iloc[:, 1:]
            
            # Modificación si se eligió clasificar por grupos
            if group:
                print("Modo 'group=True': Agrupando etiquetas (labels).")
                # Mapeo basado en Fashion-MNIST y la imagen:
                #
                # Nuevo Grupo 0 (Top): T-shirt(0), Pullover(2), Coat(4), Shirt(6)
                # Nuevo Grupo 1 (Footwear): Sandal(5), Sneaker(7), Ankle boot(9)
                # Nuevo Grupo 2 (Bottom): Trouser(1), Dress(3)
                # Nuevo Grupo 3 (Bag): Bag(8)
                label_map = {
                    0: 0, # T-shirt -> 0 (Top)
                    1: 2, # Trouser -> 2 (Bottom)
                    2: 0, # Pullover -> 0 (Top)
                    3: 2, # Dress -> 2 (Bottom)
                    4: 0, # Coat -> 0 (Top)
                    5: 1, # Sandal -> 1 (Footwear)
                    6: 0, # Shirt -> 0 (Top)
                    7: 1, # Sneaker -> 1 (Footwear)
                    8: 3, # Bag -> 3 (Bag)
                    9: 1  # Ankle boot -> 1 (Footwear)
                }
                
                # Aplicar el mapeo a la columna de etiquetas
                labels_df = labels_df.map(label_map)
                print("Mapeo de 10 clases a 4 grupos completado.")
            
            # 4. Convertir a Tensores de PyTorch
            # Las características deben ser Float (para los cálculos)
            # Las etiquetas deben ser Long (para la función de pérdida)
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_df.values, dtype=torch.long)
            
            # 5. Normalizar los datos de los píxeles (0-255 -> 0.0-1.0)
            features_tensor = features_tensor / 255.0
            
            # 6. Crear un TensorDataset
            dataset = TensorDataset(features_tensor, labels_tensor)
            
            # 7. Crear el DataLoader
            train_loader = DataLoader(dataset, batch_size=tamaño_de_lote, shuffle=True)
            
            print("DataLoader de entrenamiento creado exitosamente.")
            return train_loader
        
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    else:
        messagebox.showerror("Error en la selección de archivo", "No se seleccionó ningún archivo!")
        return None

# LISTO: Carga los datos de test en un DataLoader
def dataset_test_loader(group = False, tamaño_de_lote: int = 64, filepath: str = None):
    if filepath:
        
        print(f"Cargando datos de prueba desde: {filepath}")

        try:
            # 2. Leer el archivo CSV con Pandas
            data_df = pd.read_csv(filepath)
            
            # 3. Separar etiquetas (y) y características (X)
            labels_df = data_df.iloc[:, 0]
            features_df = data_df.iloc[:, 1:]
            
            # Modificación si se eligió clasificar por grupos
            if group:
                print("Modo 'group=True': Agrupando etiquetas (labels).")
                # Mapeo basado en Fashion-MNIST y la imagen:
                #
                # Nuevo Grupo 0 (Top): T-shirt(0), Pullover(2), Coat(4), Shirt(6)
                # Nuevo Grupo 1 (Footwear): Sandal(5), Sneaker(7), Ankle boot(9)
                # Nuevo Grupo 2 (Bottom): Trouser(1), Dress(3)
                # Nuevo Grupo 3 (Bag): Bag(8)
                label_map = {
                    0: 0, # T-shirt -> 0 (Top)
                    1: 2, # Trouser -> 2 (Bottom)
                    2: 0, # Pullover -> 0 (Top)
                    3: 2, # Dress -> 2 (Bottom)
                    4: 0, # Coat -> 0 (Top)
                    5: 1, # Sandal -> 1 (Footwear)
                    6: 0, # Shirt -> 0 (Top)
                    7: 1, # Sneaker -> 1 (Footwear)
                    8: 3, # Bag -> 3 (Bag)
                    9: 1  # Ankle boot -> 1 (Footwear)
                }
                
                # Aplicar el mapeo a la columna de etiquetas
                labels_df = labels_df.map(label_map)
                print("Mapeo de 10 clases a 4 grupos completado.")
            
            # 4. Convertir a Tensores de PyTorch
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_df.values, dtype=torch.long)
            
            # 5. Normalizar los datos de los píxeles (0-255 -> 0.0-1.0)
            features_tensor = features_tensor / 255.0
            
            # 6. Crear un TensorDataset
            dataset = TensorDataset(features_tensor, labels_tensor)
            
            # 7. Crear el DataLoader
            test_loader = DataLoader(dataset, batch_size=tamaño_de_lote, shuffle=False)
            
            print("DataLoader de prueba creado exitosamente.")
            return test_loader
        
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    else:
        return None