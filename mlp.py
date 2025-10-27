import torch
from torch import nn
from tkinter import messagebox
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, neuronas_entrada: int, neuronas_salida: int, capas_ocultas: int, neuronas_por_capa_oculta: int):
        super(MLP, self).__init__()
        
        # Guardamos las variables
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_salida = neuronas_salida
        self.capas_ocultas = capas_ocultas
        self.neuronas_por_capa_oculta = neuronas_por_capa_oculta
        
        # Capas ocultas
        self.mlp_layers = None
        
        # Dataset's
        self.train_dataset: DataLoader = None
        self.test_dataset: DataLoader = None
        
        # Flag
        self.created = False
        self.trained = False
        self.groupFashionMNIST = False
        self.rectangular = False
        self.convolucional = False
        
        # Historial de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    # LISTO: Crear capa rectangular
    def create_rectangular_network(self):
        if self.created == False:
            print("\n--- CREANDO MLP RECTANGULAR ---")

            # Creamos un for para recorrer la cantidad de capas ocultas
            layers = []
            neuronas_entrada = self.neuronas_entrada
            for i in range(self.capas_ocultas):
                # Obtenemos neuronas por c/ capa
                neuronas_salida = self.neuronas_por_capa_oculta
                
                # Creamos la capa con función de activación ReLU
                layers.append(nn.Linear(neuronas_entrada, neuronas_salida))
                
                # Añadimos la capa
                layers.append(nn.ReLU())
                
                # Actualizamos la cantidad de neuronas
                neuronas_entrada = neuronas_salida
                
                # Debug
                print(F"Capa oculta {i+1} creada. Estructura {neuronas_entrada} > {neuronas_salida}. Usa la función de activación ReLU.")
                
            # Creamos la capa de salida
            layers.append(nn.Linear(neuronas_entrada, self.neuronas_salida))
            self.mlp_layers = nn.Sequential(*layers)
            self.created = True
            self.rectangular = True
            print(F"Capa de salida creada. Estructura {neuronas_entrada} > {self.neuronas_salida}.")
    
    # LISTO: Crear capa convolucional
    def create_convolucional_network(self):
        if self.created == False:
            print("\n--- CREANDO MLP CONVOLUCIONAL ---")

            # Interpolación
            neuronas_comienzo = self.neuronas_entrada
            neuronas_termina = self.neuronas_salida
            numero_saltos = self.capas_ocultas + 1
            delta = neuronas_comienzo - neuronas_termina
            tamaño_salto = max(1, delta // numero_saltos) if delta > 0 else 0
            print(f"Interpolación {neuronas_comienzo} -> {neuronas_termina} en {numero_saltos}, paso = {tamaño_salto}")
        
            # MLP
            layers = []
            neuronas_entrada = neuronas_comienzo
            
            # Bucle que recorre la cantida de capas ocultas
            for i in range(self.capas_ocultas):
                # Calculo de neuronas de salida
                neuronas_salida = max(neuronas_termina, neuronas_entrada - tamaño_salto)
            
                # Capa oculta
                layers.append(nn.Linear(neuronas_entrada, neuronas_salida))

                # Función de activación
                layers.append(nn.ReLU())
                
                # Debug
                print(F"Capa oculta {i+1} creada. Estructura {neuronas_entrada} > {neuronas_salida}. Usa la función de activación ReLU.")

                neuronas_entrada = neuronas_salida

            # Capa salida
            layers.append(nn.Linear(neuronas_entrada, self.neuronas_salida))
            print(F"Capa de salida creada. Estructura {neuronas_entrada} > {self.neuronas_salida}.")
            
            # Actualizar arreglo
            self.mlp_layers = nn.Sequential(*layers)
            self.created = True
            self.convolucional = True
       
    # LISTO: Actualiza el dataset del modelo 
    def set_train_dataset(self, dataset):
        self.train_dataset = dataset
    
    # LISTO: Actualiza el dataset del modelo 
    def set_test_dataset(self, dataset):
        self.test_dataset = dataset
        
    # LISTO: Guardar modelo
    def save_model(self, tipo_de_red:str):
        print("\n--- GUARDANDO RED NEURONAL ---")

        # Obtener ruta de guardado
        path_and_filename = utils.seleccionar_ruta_guardado(
            "Guardar modelo IA",
            [
                ("Modelos PyTorch", "*.pth"),
                ("Todos los archivos", "*")
            ],
            ".pth"
        )
        
        if path_and_filename:
            
            # Crear el diccionario con metadatos y pesos
            model_data = {
                # Tipo de red
                'rectangular': self.rectangular,
                'convolucional': self.convolucional,
                
                # Hiperparámetros
                'neuronas_entrada': self.neuronas_entrada,
                'neuronas_salida': self.neuronas_salida,
                'capas_ocultas': self.capas_ocultas,
                'neuronas_por_capa_oculta': self.neuronas_por_capa_oculta,
                'state_dict': self.state_dict(),

                # Datos de entrenamiento
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                
                # Datos de grupo
                'group_MNIST': self.groupFashionMNIST,
                
                # Datos de los datos
                'red_entrenada': self.trained
            }
            
            # Guardar modelo en la ruta
            torch.save(model_data, path_and_filename)
            print(f"Se han actualizado los pesos de las capas!")
            
            # Debug
            print(f"Modelo guardado!")
            print(f"Ruta: {path_and_filename}")
        
        else:
            messagebox.showerror("Error de guardado", "No se eligió ninguna ruta! Se procede a cancelar el guardado")
            
    # LISTO: Carga un modelo existente
    @classmethod
    def load_model_from_file(cls, filepath:str):
        print("\n--- CARGANDO RED NEURONAL ---")
        
        # Si existe la ruta
        if filepath:
            print(f"Ruta: {filepath}")
            
            # Cargar en CPU o GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar el diccionario completo
            model_data = torch.load(filepath, map_location=device)
            
            # Arquitectura de red
            rectangular = model_data['rectangular']
            convolucional = model_data['convolucional']
            
            # Extraer los hiperparámetros guardados
            neuronas_entrada = model_data['neuronas_entrada']
            neuronas_salida = model_data['neuronas_salida']
            capas_ocultas = model_data['capas_ocultas']
            neuronas_por_capa_oculta = model_data['neuronas_por_capa_oculta']
            group_fashion_mnist = model_data['group_MNIST']
            
            # Datos de datos
            trained = model_data['red_entrenada']
            
            # Creacion del modelo
            model: MLP = cls(
                neuronas_entrada,
                neuronas_salida,
                capas_ocultas,
                neuronas_por_capa_oculta
            )
            
            # Salvamos el tipo de clasificacion MNIST
            model.groupFashionMNIST = group_fashion_mnist
            
            # Salvamos el dato de red entrenada
            model.trained = trained
            
            # Construir la arquitectura correcta
            if rectangular:
                model.create_rectangular_network()
            elif convolucional:
                model.create_convolucional_network()
            else:
                raise ValueError(f"Tipo de modelo desconocido en el archivo")
            
            # Cargar los pesos (state_dict) en el modelo recién creado
            model.load_state_dict(model_data['state_dict'])
            print("Pesos actualizados")
            
            # Cargamos historial
            model.train_losses = model_data.get('train_losses', [])
            model.val_losses = model_data.get('val_losses', [])
            model.val_accuracies = model_data.get('val_accuracies', [])
            
            # Mostramos info
            print("\n--- INICIANDO CREACIÓN DE RED ---")
            print(f"Tipo de red: [Convolucional = {convolucional}] | [Rectangular = {rectangular}]")
            print(f"El modelo clasifica grupo: {group_fashion_mnist}")
            print(f"Neuronas entrada: {neuronas_entrada}")
            print(f"Neuronas salida: {neuronas_salida}")
            print(f"Capas ocultas: {capas_ocultas}")
            
            if rectangular == True:
                print(f"Neuronas por capa: {neuronas_por_capa_oculta}")
            
            # Poner en modo evaluación
            model.eval()
            
            # Debug
            print(f"¡Modelo cargado exitosamente desde {filepath}!")
            messagebox.showinfo("Carga exitosa", "El modelo se ha cargado correctamente.")
            
            # Retornamos la instancia del modelo cargado
            return model 
        
    # LISTO: Entrena el modelo
    def train_model(self, optimizador: str, tasa_aprendizaje: float, epocas: int, continue_training: bool = False):
        print("\n--- CONFIGURANDO ENTRENAMIENTO ---")
        
        # Config entrenamiento
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {device}")
        
        # Mueve el modelo (self) al dispositivo
        self.to(device) 
        
        # Criterio a usar
        criterion = nn.CrossEntropyLoss()
        
        # Optimizador a usar
        optimizer_instance = None
        if optimizador.lower() == 'adam':
            optimizer_instance = torch.optim.Adam(self.parameters(), lr=tasa_aprendizaje)
        
        elif optimizador.lower() == 'sgd':
            optimizer_instance = torch.optim.SGD(self.parameters(), lr=tasa_aprendizaje, momentum=0.9)
        
        elif optimizador.lower() == 'rmsprop':
            optimizer_instance = torch.optim.RMSprop(self.parameters(), lr=tasa_aprendizaje, momentum=0.9)
        
        else:
            print(f"Optimizador '{optimizador}' no reconocido. Usando 'Adam' por defecto.")
            optimizer_instance = torch.optim.Adam(self.parameters(), lr=tasa_aprendizaje)

        print(f"Optimizador: {optimizador.lower()}, Tasa de Aprendizaje: {tasa_aprendizaje}, Épocas: {epocas}")
    
        # print("\n-------------------------------------------------------------")
        # print(f"DEBUG: Variable self.trained: {self.trained}")
        # print("-------------------------------------------------------------\n")
        # El usuario explícitamente eligió "No" para continuar, o es la primera vez.
        if not continue_training:
            self.clear_history() 
            print("Historial de entrenamiento limpiado (Inicio nuevo).")
        else:
            # El usuario eligió "Sí" para continuar.
            # Ahora, verificamos si de verdad hay algo que continuar.
            if not self.trained:
                # Caso raro: El usuario dijo "Sí" pero el flag .trained es Falso
                self.clear_history()
                print("Advertencia: Se pidió continuar, pero no hay historial previo. Iniciando de cero.")
            else:
                # Caso exitoso: El usuario dijo "Sí" y hay historial.
                print("\n--- CONTINUANDO ENTRENAMIENTO ---")
                print("El historial de pérdidas y precisión anterior se conservará.")
                # Bucle de Entrenamiento y Validación
                print("\n--- INICIANDO ENTRENAMIENTO ---")
        
        # Capturamos la barra de épocas en una variable
        bar_epocas = tqdm(range(epocas), desc="Progreso Total", unit="epoch")
        
        for epoch in bar_epocas:
            
            # Fase entrenamiento
            self.train() 
            running_train_loss = 0.0
            
            # Envolvemos el dataloader de train en tqdm
            train_bar = tqdm(self.train_dataset, desc=f"Epoch {epoch+1} [Train]", leave=False)
            
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer_instance.zero_grad()
                
                # Llama al método forward()
                outputs = self(inputs) 
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_instance.step()
                running_train_loss += loss.item()

                # Actualiza la barra INTERNA con la pérdida actual
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Fase de Validación (Test)
            self.eval() 
            running_val_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            # Envolvemos el dataloader de test en tqdm
            val_bar = tqdm(self.test_dataset, desc=f"Epoch {epoch+1} [Valid]", leave=False)
            
            # Desactivar el cálculo de gradientes
            with torch.no_grad(): 
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Llama al método forward()
                    outputs = self(inputs) 
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    
                    # Calcular la precisión
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    # Actualiza la barra INTERNA con la pérdida de validación
                    val_bar.set_postfix(val_loss=f"{loss.item():.4f}")

            # Guardar y Mostrar Métricas de la Época
            avg_train_loss = running_train_loss / len(self.train_dataset)
            avg_val_loss = running_val_loss / len(self.test_dataset)
            accuracy = 100 * correct_predictions / total_samples
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(accuracy)

            # Actualizamos la barra de épocas
            metrics = {
                'Train Loss': f"{avg_train_loss:.4f}",
                'Val Loss': f"{avg_val_loss:.4f}",
                'Val Acc': f"{accuracy:.2f}%"
            }
            bar_epocas.set_postfix(metrics)

        # Añadimos un print vacío para asegurar que el cursor baje
        print() 
        print("--- ENTRENAMIENTO FINALIZADO ---")
        self.trained = True
        messagebox.showinfo("Entrenamiento Exitoso", f"Modelo entrenado durante {epocas} épocas.\nPrecisión final: {self.val_accuracies[-1]:.2f}%")
    
    # LISTO: Realiza el forward
    def forward(self, x):
        if not self.created or self.mlp_layers is None:
            raise RuntimeError("Error fatal! Primero debe crear la red")
        
        return self.mlp_layers(x)
    
    # LISTO: Limpia el historial de entrenamiento
    def clear_history(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    # LISTO: Prueba el modelo con el dataset de prueba. Devuelve predc. y etiquetas
    def test_model(self):
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Modo evaluación
        self.eval() 
        
        all_labels = []
        all_preds = []
        
        # Iterar sin gradientes
        with torch.no_grad():
            # Usamos tqdm para la consola
            test_bar = tqdm(self.test_dataset, desc="[MLP] Evaluando en Test", leave=False)
            
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(inputs)
                
                # Obtener la predicción (la clase con el logit más alto)
                _, predicted = torch.max(outputs.data, 1)
                
                # Guardar en listas (moviendo a CPU para .numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(f"[MLP] Evaluación completada. Total de muestras: {len(all_labels)}")
        
        # Devolver listas
        return all_preds, all_labels
    
    