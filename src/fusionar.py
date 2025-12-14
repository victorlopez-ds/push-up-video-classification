import nbformat
import os
import re

# 1. Lista exacta de tus archivos en orden
archivos = [
    "push-up-classification-eda.ipynb",
    "baseline.ipynb",
    "modeladoLSTM_Simple_sin_fugas.ipynb",
    "ST-GCN_model.ipynb",
    "results-analysis.ipynb"
]

nombre_salida = "entrega-push-ups-classification.ipynb"

# --- FUNCIONES PARA GENERAR ÍNDICE ---
def limpiar_anchor(texto):
    """Convierte un título en un enlace compatible con Jupyter (kebab-case)"""
    texto = texto.lower()
    texto = texto.replace(" ", "-")
    # Eliminar cualquier carácter que no sea letra, número o guion
    texto = re.sub(r'[^a-z0-9\-]', '', texto)
    return texto

def generar_indice(notebook):
    """Escanea el notebook y crea una celda con la Tabla de Contenidos"""
    toc_md = ["# 📑 Índice del Proyecto\n"]
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            lines = cell.source.split('\n')
            for line in lines:
                line = line.strip()
                # Detectar encabezados (#, ##, ###)
                if line.startswith('#'):
                    # Contar nivel de profundidad (h1, h2, etc.)
                    nivel = len(line.split(' ')[0]) 
                    titulo = line.lstrip('#').strip()
                    
                    if titulo:
                        anchor = limpiar_anchor(titulo)
                        # Indentación visual para sub-apartados
                        indent = "    " * (max(0, nivel - 1))
                        # Formato Markdown: [Título](#enlace)
                        toc_md.append(f"{indent}- [{titulo}](#{anchor})")
    
    if len(toc_md) > 1:
        return nbformat.v4.new_markdown_cell("\n".join(toc_md))
    return None

# --- INICIO DEL PROCESO ---

# 2. Crear el notebook base vacío
notebook_final = nbformat.v4.new_notebook()

print(f"--- Iniciando fusión de {len(archivos)} notebooks ---")

# 3. Proceso de fusión
for nombre_archivo in archivos:
    if not os.path.exists(nombre_archivo):
        print(f"❌ ERROR: No encuentro el archivo: {nombre_archivo}")
        continue
    
    try:
        # Leemos el archivo asegurando codificación utf-8
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Añadimos las celdas al notebook final
        notebook_final.cells.extend(nb.cells)
        print(f"✅ Añadido: {nombre_archivo}")
        
    except Exception as e:
        print(f"⚠️ Error leyendo {nombre_archivo}: {e}")

# 4. Generar e Insertar Índice
print("--- Generando Índice ---")
celda_indice = generar_indice(notebook_final)

if celda_indice:
    # Insertamos la celda en la posición 0 (al principio)
    notebook_final.cells.insert(0, celda_indice)
    print("✅ Índice creado e insertado al inicio.")
else:
    print("⚠️ No se encontraron títulos para crear el índice.")

# 5. Guardar el resultado
try:
    with open(nombre_salida, 'w', encoding='utf-8') as f:
        nbformat.write(notebook_final, f)
    print(f"\n🎉 ¡ÉXITO! Archivo creado correctamente: {nombre_salida}")
except Exception as e:
    print(f"\n❌ Error guardando el archivo final: {e}")