# Clasificación de Técnica en Flexiones (Push-ups) con Visión Artificial

⚠️ **Trabajo de Fin de Asignatura - Aprendizaje Automático Avanzado**

Este proyecto explora el uso de técnicas de Deep Learning y Visión Artificial para analizar y clasificar la calidad de la ejecución en ejercicios de fuerza, específicamente flexiones (push-ups).

## 🚀 Descripción del Proyecto

El objetivo es desarrollar un sistema capaz de distinguir entre una técnica correcta e incorrecta analizando secuencias de vídeo. En lugar de procesar los píxeles crudos, utilizamos **MediaPipe** para extraer la pose corporal (landmarks) y modelar la biomecánica del movimiento.

### Tecnologías Clave
- **MediaPipe Pose:** Para extracción de esqueletos y normalización.
- **ST-GCN (Spatial-Temporal Graph Convolutional Networks):** Modelado del cuerpo como un grafo para capturar relaciones espaciales y temporales.
- **LSTM / Bi-LSTM:** Análisis de secuencias temporales.
- **PyTorch:** Framework de Deep Learning.

## 📂 Estructura del Repositorio

```
.
├── data/               # Dataset de vídeos (originales y procesados)
├── docs/               # Documentación y enunciados del proyecto
├── img/                # Imágenes y recursos gráficos
├── notebooks/          # Notebooks principales del proyecto
│   ├── entrega-push-ups-classification.ipynb  # INFORME FINAL y Demostración
│   ├── ST-GCN_model.ipynb                     # Implementación y entrenamiento ST-GCN
│   ├── modeladoLSTM_Simple_sin_fugas.ipynb    # Experimentos con LSTM
│   ├── EDA.ipynb                              # Análisis Exploratorio de Datos
│   └── archive/                               # Experimentos preliminares y versiones antiguas
├── results/            # Modelos entrenados y métricas
├── src/                # Código fuente y utilidades
│   └── utils.py        # Funciones auxiliares (procesamiento video, landmarks, etc.)
└── README.md           # Este archivo
```

## 🛠️ Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/victorlopez-ds/push-up-video-classification.git
    cd push-up-video-classification
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar los notebooks:**
    Se recomienda empezar por `notebooks/entrega-push-ups-classification.ipynb` para una visión general del proyecto y resultados.

## 👥 Autores

- **Víctor López**
- **Marcos Yordanov Marín**

Grado en Ciencia e Ingeniería de Datos - Universidad de Murcia.
