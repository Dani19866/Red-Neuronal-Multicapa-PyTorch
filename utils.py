from tkinter import filedialog
from tkinter import messagebox

def read_csv(path):
    vectores = []
    salidas = []
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            
            # Todo menos x1, x2, ..., xn, y (encabezado)
            for line in lines[1:]:
                aux = line.strip().split(",")
                valores_numericos = [float(v) for v in aux]

                # Las primeras n-1 columnas son las entradas
                vectores.append(valores_numericos[:-1])

                # La última columna es la salida
                salidas.append(int(valores_numericos[-1]))
                
                # Debug
                # print(f"Leído vector: {valores_numericos[:-1]} | Salida: {valores_numericos[-1]}")
        
        return vectores, salidas
    except:
        messagebox.showerror("Error de lectura", "No se pudo leer el archivo CSV seleccionado.")
        return -1, -1
        
def seleccionar_archivo(title, filetypes):
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return filepath if filepath else None

def seleccionar_ruta_guardado(title, filetypes, defaultextension):
    filepath = filedialog.asksaveasfilename(title=title, filetypes=filetypes, defaultextension=defaultextension)
    return filepath if filepath else None