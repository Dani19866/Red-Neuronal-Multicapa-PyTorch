import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import gui_functions  # Asumiendo que este módulo existe
import utils  # Asumiendo que este módulo existe
from mlp import MLP

class ConfiguradorRedGUI(tk.Tk):
    mlp: MLP = None

    def __init__(self):
        super().__init__()

        # --- Configuración de la ventana principal ---
        self.title("Configurador de Red Neuronal")
        self.geometry("480x525")
        self.resizable(False, False)

        # --- Variables de control de Tkinter (Pestaña 1) ---
        self.modo_creacion = tk.StringVar(value="crear")
        self.ruta_archivo = tk.StringVar(value="")
        self.tipo_red = tk.StringVar(value="rectangular")
        self.tipo_clasificacion = tk.StringVar(value="mnist")
        self.neuronas_entrada = tk.StringVar(value="784")
        self.neuronas_salida = tk.StringVar(value="10")
        self.capas_ocultas = tk.StringVar(value="2")
        self.neuronas_por_capa = tk.StringVar(value="512")

        # --- Variables de control de Tkinter (Pestaña 2) ---
        self.ruta_entrenamiento = tk.StringVar(value="")
        self.ruta_test = tk.StringVar(value="")
        self.ruta_entrenamiento_display = tk.StringVar(value="No seleccionado")
        self.ruta_test_display = tk.StringVar(value="No seleccionado")
        self.optimizador = tk.StringVar(value="Adam")
        self.tasa_aprendizaje = tk.StringVar(value="0.001")
        self.epocas = tk.StringVar(value="10")
        self.tamaño_de_lote = tk.StringVar(value="64")

        # --- Registros de validación ---
        self.vcmd_numeric = (self.register(self._validar_solo_numeros), '%P')
        self.vcmd_float = (self.register(self._validar_solo_floats), '%P')

        # --- Crear la interfaz ---
        
        # 1. Crear el control de pestañas
        self.notebook = ttk.Notebook(self)
        
        # 2. Llamar a los métodos para construir cada pestaña, pasando el notebook
        self._crear_tab1(self.notebook)
        self._crear_tab2(self.notebook)

        # 3. Empaquetar el Notebook al final
        self.notebook.pack(expand=True, fill="both")
        
        # 4. Vincular eventos y actualizar estado
        self._vincular_eventos()
        self._actualizar_estado_widgets()

    # -------------------- Pestañas ------------------------------
    
    # LISTO: Creación de la primera pestaña
    def _crear_tab1(self, notebook):
        
        # Crear el Frame de la pestaña y añadirlo al notebook
        self.tab1 = ttk.Frame(notebook, padding=15)
        notebook.add(self.tab1, text="Creación de la red")

        # --- Contenedor 1: Selección básica ---
        self.lf_seleccion = ttk.LabelFrame(self.tab1, text="1. Selección básica")
        self.lf_seleccion.pack(fill="x", padx=10, pady=5)
        
        self.rb_crear = ttk.Radiobutton(self.lf_seleccion, text="Crear nueva red", 
                                          variable=self.modo_creacion, value="crear")
        self.rb_crear.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.rb_cargar = ttk.Radiobutton(self.lf_seleccion, text="Cargar red", 
                                           variable=self.modo_creacion, value="cargar")
        self.rb_cargar.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        self.btn_seleccionar_archivo = ttk.Button(self.lf_seleccion, text="Seleccionar archivo...",
                                                    command=self._seleccionar_archivo)
        self.btn_seleccionar_archivo.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # --- Contenedor 2: Parámetros generales ---
        self.lf_parametros = ttk.LabelFrame(self.tab1, text="2. Parámetros generales")
        self.lf_parametros.pack(fill="x", padx=10, pady=10)

        # (Widgets de Contenedor 2...)
        ttk.Label(self.lf_parametros, text="Tipo de red:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.rb_rect = ttk.Radiobutton(self.lf_parametros, text="Rectangular (MLP)", 
                                         variable=self.tipo_red, value="rectangular")
        self.rb_rect.grid(row=0, column=1, sticky="w", padx=5)
        self.rb_conv = ttk.Radiobutton(self.lf_parametros, text="Convolucional (CNN)", 
                                         variable=self.tipo_red, value="convolucional")
        self.rb_conv.grid(row=0, column=2, sticky="w", padx=5)
        ttk.Label(self.lf_parametros, text="Clasificación:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.rb_mnist = ttk.Radiobutton(self.lf_parametros, text="Fashion MNIST (10)", 
                                          variable=self.tipo_clasificacion, value="mnist")
        self.rb_mnist.grid(row=1, column=1, sticky="w", padx=5)
        self.rb_articulos = ttk.Radiobutton(self.lf_parametros, text="Grupo de artículos (4)", 
                                              variable=self.tipo_clasificacion, value="articulos")
        self.rb_articulos.grid(row=1, column=2, sticky="w", padx=5)

        # --- Contenedor 3: Parámetros de la red ---
        self.lf_arquitectura = ttk.LabelFrame(self.tab1, text="3. Parámetros de la red")
        self.lf_arquitectura.pack(fill="x", padx=10, pady=10)
        
        # (Widgets de Contenedor 3...)
        ttk.Label(self.lf_arquitectura, text="Neuronas de entrada:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.entry_entrada = ttk.Entry(self.lf_arquitectura, textvariable=self.neuronas_entrada, 
                                         state="disabled", width=10)
        self.entry_entrada.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(self.lf_arquitectura, text="Neuronas de salida:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.entry_salida = ttk.Entry(self.lf_arquitectura, textvariable=self.neuronas_salida, 
                                        state="disabled", width=10)
        self.entry_salida.grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(self.lf_arquitectura, text="Número de capas ocultas:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.entry_capas = ttk.Entry(self.lf_arquitectura, textvariable=self.capas_ocultas,
                                       validate="key", validatecommand=self.vcmd_numeric, width=10)
        self.entry_capas.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(self.lf_arquitectura, text="Neuronas por capa (MLP):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.entry_neuronas = ttk.Entry(self.lf_arquitectura, textvariable=self.neuronas_por_capa,
                                          validate="key", validatecommand=self.vcmd_numeric, width=10)
        self.entry_neuronas.grid(row=3, column=1, padx=10, pady=5)

        # --- Contenedor 4: Finalizar red ---
        lf_finalizar = ttk.LabelFrame(self.tab1, text="4. Finalizar red")
        lf_finalizar.pack(fill="x", padx=10, pady=10)

        self.btn_crear_cargar = ttk.Button(lf_finalizar, text="Crear / Cargar Red", 
                                             command=self._crear_cargar_red)
        self.btn_crear_cargar.pack(pady=10, padx=20, fill="x")


    # LISTO: Creación de la segunda pestaña
    def _crear_tab2(self, notebook):
        """Crea todos los widgets para la Pestaña 2."""

        # Crear el Frame de la pestaña y añadirlo al notebook
        self.tab2 = ttk.Frame(notebook, padding=15)
        notebook.add(self.tab2, text="Entrenamiento y pruebas")
        
        # --- Contenedor 1: Selección de datasets ---
        lf_datasets = ttk.LabelFrame(self.tab2, text="1. Selección de datasets")
        lf_datasets.pack(fill="x", padx=10, pady=10)
        lf_datasets.columnconfigure(1, weight=1)

        # (Widgets de Contenedor 1...)
        ttk.Label(lf_datasets, text="Opción 1: Archivos locales").grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(5,0))
        self.btn_train = ttk.Button(lf_datasets, text="Archivo de entrenamiento", command=self._seleccionar_entrenamiento)
        self.btn_train.grid(row=1, column=0, sticky="ew", padx=(15, 5), pady=5)
        self.lbl_train = ttk.Label(lf_datasets, textvariable=self.ruta_entrenamiento_display, relief="sunken", anchor="w", padding=2)
        self.lbl_train.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=5)
        self.btn_test = ttk.Button(lf_datasets, text="Archivo de test", command=self._seleccionar_test)
        self.btn_test.grid(row=2, column=0, sticky="ew", padx=(15, 5), pady=5)
        self.lbl_test = ttk.Label(lf_datasets, textvariable=self.ruta_test_display, relief="sunken", anchor="w", padding=2)
        self.lbl_test.grid(row=2, column=1, sticky="ew", padx=(5, 10), pady=5)

        # --- Contenedor 2: Opciones de entrenamiento ---
        lf_opciones = ttk.LabelFrame(self.tab2, text="2. Opciones de entrenamiento")
        lf_opciones.pack(fill="x", padx=10, pady=10)

        # (Widgets de Contenedor 2...)
        ttk.Label(lf_opciones, text="Optimizador:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.combo_opt = ttk.Combobox(lf_opciones, textvariable=self.optimizador, values=["Adam", "SGD", "RMSprop"], state="readonly", width=15)
        self.combo_opt.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(lf_opciones, text="Tasa de aprendizaje:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.entry_lr = ttk.Entry(lf_opciones, textvariable=self.tasa_aprendizaje, validate="key", validatecommand=self.vcmd_float, width=17)
        self.entry_lr.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(lf_opciones, text="Número de épocas:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.entry_epocas = ttk.Entry(lf_opciones, textvariable=self.epocas, validate="key", validatecommand=self.vcmd_numeric, width=17)
        self.entry_epocas.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(lf_opciones, text="Tamaño de lote (batch size):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.entry_epocas = ttk.Entry(lf_opciones, textvariable=self.tamaño_de_lote, validate="key", validatecommand=self.vcmd_numeric, width=17)
        self.entry_epocas.grid(row=3, column=1, sticky="w", padx=10, pady=5)

        # --- Contenedor 3: Acciones ---
        lf_acciones = ttk.LabelFrame(self.tab2, text="3. Acciones")
        lf_acciones.pack(fill="x", padx=10, pady=10)
        
        # (Widgets de Contenedor 3...)
        lf_acciones.columnconfigure(0, weight=1)
        lf_acciones.columnconfigure(1, weight=1)
        lf_acciones.columnconfigure(2, weight=1)
        self.btn_entrenar = ttk.Button(lf_acciones, text="Entrenar Red", command=self._entrenar_red)
        self.btn_entrenar.grid(row=0, column=0, sticky="ew", padx=5, pady=10)
        self.btn_probar = ttk.Button(lf_acciones, text="Probar Red", command=self._probar_red)
        self.btn_probar.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        self.btn_guardar = ttk.Button(lf_acciones, text="Guardar Red", command=self._guardar_red)
        self.btn_guardar.grid(row=0, column=2, sticky="ew", padx=5, pady=10)

        # --- Contenedor 4: Métricas ---
        lf_metricas = ttk.LabelFrame(self.tab2, text="4. Métricas")
        lf_metricas.pack(fill="x", padx=10, pady=10)

        # (Widgets de Contenedor 4...)
        lf_metricas.columnconfigure(0, weight=1)
        lf_metricas.columnconfigure(1, weight=1)
        lf_metricas.columnconfigure(2, weight=1)
        self.btn_loss = ttk.Button(lf_metricas, text="Ver Pérdida (Loss)", command=self._mostrar_loss)
        self.btn_loss.grid(row=0, column=0, sticky="ew", padx=5, pady=10)
        self.btn_matriz = ttk.Button(lf_metricas, text="Matriz de Confusión", command=self._mostrar_matriz)
        self.btn_matriz.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        self.btn_precision = ttk.Button(lf_metricas, text="Métricas de Precisión", command=self._mostrar_precisiones)
        self.btn_precision.grid(row=0, column=2, sticky="ew", padx=5, pady=10)

    # -------------------- Utilidades ----------------------------
    
    # LISTO: Vincular eventos con funciones. Todos los eventos son WRITE
    def _vincular_eventos(self):
        self.modo_creacion.trace_add("write", self._actualizar_estado_widgets)
        self.tipo_red.trace_add("write", self._actualizar_estado_widgets)
        self.tipo_clasificacion.trace_add("write", self._actualizar_estado_widgets)
    
    # LISTO: Actualiza el estado de los componentes Tkinter
    def _actualizar_estado_widgets(self, *args):
        modo = self.modo_creacion.get()
        if modo == "crear":
            self.btn_seleccionar_archivo.config(state="disabled")
            self._set_estado_contenedor(self.lf_parametros, "normal")
            self._set_estado_contenedor(self.lf_arquitectura, "normal")
            self.entry_entrada.config(state="disabled")
            self.entry_salida.config(state="disabled")
            if self.tipo_clasificacion.get() == "mnist":
                self.neuronas_salida.set("10")
            else:
                self.neuronas_salida.set("4")
            if self.tipo_red.get() == "convolucional":
                self.entry_neuronas.config(state="disabled")
            else:
                self.entry_neuronas.config(state="normal")
        elif modo == "cargar":
            self.btn_seleccionar_archivo.config(state="normal")
            self._set_estado_contenedor(self.lf_parametros, "disabled")
            self._set_estado_contenedor(self.lf_arquitectura, "disabled")

    # LISTO: Habilitar/deshabilitar widgets
    def _set_estado_contenedor(self, contenedor, estado):
        # (Sin cambios)
        for widget in contenedor.winfo_children():
            try:
                widget.config(state=estado)
            except tk.TclError:
                pass
    
    # LISTO: Pasa de pestaña 1 a pestaña 2
    def _activar_tab2(self):
        try:
            self.notebook.select(self.tab2)
            self.notebook.tab(self.tab1, state="disabled")
        except tk.TclError as e:
            print(f"Error al cambiar de pestaña: {e}")
            messagebox.showerror("Error de UI", "No se pudo cambiar a la pestaña 2.")

    # -------------------- Validaciones --------------------------
    
    # LISTO: Valida formato de int
    def _validar_solo_numeros(self, P):
        # (Sin cambios)
        if P == "" or P.isdigit():
            return True
        return False

    # LISTO: Valida formato de float
    def _validar_solo_floats(self, P):
        # (Sin cambios)
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            if P == ".":
                return True
            return False

    # -------------------- Pestaña 1: Eventos --------------------
    
    # LISTO: Crea/carga una red neuronal
    def _crear_cargar_red(self):
        modo = self.modo_creacion.get()
        
        if modo == "cargar":
            if not self.ruta_archivo.get():
                messagebox.showerror("Error", "No se ha seleccionado ningún archivo para cargar.")
                return
            
            print("--- INICIANDO CARGA DE RED ---")
            print(f"Ruta del archivo: {self.ruta_archivo.get()}")
            self.mlp = gui_functions.load_model(self.ruta_archivo.get())
            
            if self.mlp:
                self._activar_tab2()
            
        elif modo == "crear":
            if not self.capas_ocultas.get() or \
               (self.tipo_red.get() == "rectangular" and not self.neuronas_por_capa.get()):
                messagebox.showerror("Error", "Debe rellenar todos los campos numéricos.")
                return

            print("--- INICIANDO CREACIÓN DE RED ---")
            print(f"Tipo de red: {self.tipo_red.get()}")
            print(f"Clasificación: {self.tipo_clasificacion.get()}")
            print(f"Neuronas entrada: {self.neuronas_entrada.get()}")
            print(f"Neuronas salida: {self.neuronas_salida.get()}")
            print(f"Capas ocultas: {self.capas_ocultas.get()}")
            
            agrupar_articulos = False
            if self.tipo_clasificacion.get() == 'articulos':
                agrupar_articulos = True
            
            if self.tipo_red.get() == "rectangular":
                print(f"Neuronas por capa: {self.neuronas_por_capa.get()}")
                self.mlp = gui_functions.create_model(
                    tipo_de_red=self.tipo_red.get(),
                    neuronas_entrada=self.neuronas_entrada.get(),
                    neuronas_salida=self.neuronas_salida.get(),
                    capas_ocultas=self.capas_ocultas.get(),
                    neuronas_por_capa=self.neuronas_por_capa.get(),
                    agrupar_articulos=agrupar_articulos
                )
                print("\n--- STATUS DE LA RED ---")
                print("Red MLP rectangular creada exitosamente!")
                print(f"Se va a entrenar la red para que aprenda a clasificar: {self.tipo_clasificacion.get()}")     
            else:
                self.mlp = gui_functions.create_model(
                    tipo_de_red=self.tipo_red.get(),
                    neuronas_entrada=self.neuronas_entrada.get(),
                    neuronas_salida=self.neuronas_salida.get(),
                    capas_ocultas=self.capas_ocultas.get(),
                    agrupar_articulos=agrupar_articulos
                )
                print("\n--- STATUS DE LA RED ---")
                print("Red convolucional rectangular creada exitosamente!")   
                print(f"Se va a entrenar la red para que aprenda a clasificar: {self.tipo_clasificacion.get()}")
            
            if self.mlp:
                self._activar_tab2()

    # LISTO: Selecciona el archivo que contiene el modelo de la red neuronal
    def _seleccionar_archivo(self):
        # (Sin cambios)
        filepath = utils.seleccionar_archivo(
            "Seleccionar modelo de red neuronal",
            [("Archivo PyTorch", "*pth"), ("Todos los archivos", "*")]
        )
        if filepath:
            self.ruta_archivo.set(filepath)
            print(f"Archivo seleccionado: {filepath}")

    # -------------------- Pestaña 2: Eventos --------------------

    # LISTO: Selecciona el archivo de entrenamiento
    def _seleccionar_entrenamiento(self):
        filepath = utils.seleccionar_archivo("Seleccionar archivo de entrenamiento", [("Archivos CSV", "*.csv"), ("Todos los archivos", "*")])
        if filepath:
            self.ruta_entrenamiento.set(filepath)
            filename = filepath.split('/')[-1]
            self.ruta_entrenamiento_display.set(filename)

    # LISTO: Selecciona el archivo de test
    def _seleccionar_test(self):
        filepath = utils.seleccionar_archivo("Seleccionar archivo de test", [("Archivos CSV", "*.csv"), ("Todos los archivos", "*")])
        if filepath:
            self.ruta_test.set(filepath)
            filename = filepath.split('/')[-1]
            self.ruta_test_display.set(filename)

    # LISTO: Entrena la red. Llama a gui_functions.create_model()
    def _entrenar_red(self):
        print("\n--- ACCIÓN: Entrenar Red ---")
        if not self.epocas.get() or not self.tasa_aprendizaje.get():
             messagebox.showerror("Error", "Debe definir Épocas y Tasa de Aprendizaje.")
             return
         
        # Ejecutamos la función de entrenamiento
        gui_functions.train_model(
            mlp=self.mlp, 
            optimizador=self.optimizador.get(),
            tasa_aprendizaje=self.tasa_aprendizaje.get(), 
            epocas=self.epocas.get(),
            train_dataset_path=self.ruta_entrenamiento.get(), 
            test_dataset_path=self.ruta_test.get(),
            tipo_de_clasificacion=self.tipo_clasificacion.get(),
            tamaño_de_lote=self.tamaño_de_lote.get()
            ) 
        
    # LISTO: Prueba la red. Llama a gui_functions.test_model()
    def _probar_red(self):
        print("--- ACCIÓN: Probar Red ---")
        gui_functions.test_model(self.mlp, new_path=True, tamaño_de_lote=self.tamaño_de_lote.get())

    # LISTO: Guarda la red. Llama a gui_functions.save_model()
    def _guardar_red(self):
        print("\n--- ACCIÓN: Guardar Red ---")
        gui_functions.save_model(self.mlp, self.tipo_red.get())
        
    # LISTO: Muestra el gráfico de pérdida por época. Llama a gui_functions.loss_graphic()
    def _mostrar_loss(self):
        print("--- ACCIÓN: Mostrar Pérdida (Loss) ---")
        if not self.mlp:
            messagebox.showwarning("Advertencia", "No hay un modelo entrenado para mostrar métricas.")
            return

        gui_functions.loss_graphic(self.mlp)
    
    # LISTO: Muestra la matriz de confusión. Llama a gui_functions.confusion_matrix()
    def _mostrar_matriz(self):
        print("--- ACCIÓN: Mostrar Matriz de Confusión ---")
        if not self.mlp:
            messagebox.showwarning("Advertencia", "No hay un modelo entrenado para mostrar métricas.")
            return
        
        gui_functions.confusion_matrix(self.mlp)

    # LISTO: Muestra las precisiones del entrenamiento. Llama a gui_functions.precision_metrics()
    def _mostrar_precisiones(self):
        print("--- ACCIÓN: Mostrar Métricas de Precisión ---")
        if not self.mlp:
            messagebox.showwarning("Advertencia", "No hay un modelo entrenado para mostrar métricas.")
            return
        
        gui_functions.precision_metrics(self.mlp)

if __name__ == "__main__":
    app = ConfiguradorRedGUI()
    app.mainloop()