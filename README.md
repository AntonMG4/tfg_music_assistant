# 🎵 Music Assistant based on LLMs & NLP

> **Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Informática**
> *Escuela Técnica Superior de Ingeniería (ETSI), Universidad de Huelva*
>
> **Autor:** Antón Maestre Gómez | **Tutor:** Jacinto Mata Vázquez

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Colab](https://img.shields.io/badge/Google-Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
[![Framework](https://img.shields.io/badge/DialogueKit-Flask-green)](https://github.com/iai-group/DialogueKit)
[![Frontend](https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react&logoColor=black)](https://react.dev/)
([https://img.shields.io/badge/PEFT-LoRA-red](https://img.shields.io/badge/PEFT-LoRA-red))]([https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index))
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)]([https://opensource.org/licenses/Apache-2.0](https://opensource.org/licenses/Apache-2.0))

## 📄 Descripción del Proyecto

Este proyecto presenta el desarrollo de un **asistente musical conversacional** capaz de interpretar peticiones en lenguaje natural y gestionar una lista de reproducción en tiempo real.

A diferencia de los asistentes tradicionales basados en comandos rígidos, este sistema utiliza **Modelos de Lenguaje Generativos (LLMs)** ajustados mediante técnicas de *Fine-Tuning* para clasificar la intención del usuario. El sistema actúa como un orquestador inteligente que traduce lenguaje natural en operaciones SQL sobre una base de datos musical.

### Funcionalidades Principales
El modelo clasifica cada interacción del usuario en una de las cuatro intenciones soportadas:
*   ✅ **Add:** Añadir una canción específica a la playlist (ej: *"Pon 'Bohemian Rhapsody' de Queen"*).
*   ❌ **Remove:** Eliminar una canción concreta (ej: *"Quita esa canción de la lista"*).
*   👀 **View:** Consultar el estado actual de la lista de reproducción.
*   🗑️ **Clear:** Vaciar la lista completa de golpe.

---

## 🧠 Modelos y Metodología

Para este trabajo se han comparado, optimizado y desplegado dos arquitecturas de *State-of-the-Art*:

| Modelo | Parámetros | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| **LLaMA-3.2-1B-Instruct** | 1.2B | Meta | Modelo ligero optimizado para seguir instrucciones. |
| **Falcon-7B** | 7B | TII | Modelo entrenado en el corpus masivo RefinedWeb. |

### 🔬 Entrenamiento y Optimización
*   **Dataset:** 1600 frases sintéticas generadas con ChatGPT, balanceadas perfectamente entre las 4 clases (400 ejemplos/clase).
*   **Fine-Tuning:** Se utilizó **LoRA (Low-Rank Adaptation)** para reentrenar los modelos en GPUs T4 de Google Colab, reduciendo drásticamente el consumo de VRAM.
*   **Hiperparámetros:** Ajustados mediante Optimización Bayesiana con **Optuna**.

### 📊 Resultados
Ambos modelos alcanzaron una **Exactitud (Accuracy) del 86.3%** en el conjunto de test, superando ampliamente a los *baselines* zero-shot (69-75%).

| Métrica | LLaMA-1B (Tuned) | Falcon-7B (Tuned) |
| :--- | :--- | :--- |
| **Accuracy** | **86.3%** | **86.3%** |
| **Precision** | 0.87 | 0.89 |
| **Recall** | 0.86 | 0.86 |
| **F1-Score** | 0.86 | 0.86 |

---

## 📂 Estructura del Repositorio

El proyecto se divide en tres módulos principales: **Entrenamiento** (Notebooks), **Backend** (Lógica del Agente) y **Frontend** (Interfaz Web).

.
├── chatwidget/                   # 🎨 FRONTEND (React & TypeScript)
│   ├── src/                      # Componentes del chat y lógica de UI
│   ├── public/                   # Assets estáticos
│   ├── package.json              # Dependencias de Node.js
│   ├── music_recommender.py      # 🤖 BACKEND (Orquestador DialogueKit)
│   └── chatwidget.md             # Documentación específica del widget
│
├── colab_notebooks/              # 📓 ENTRENAMIENTO E INFERENCIA
│   ├── data/                     # Dataset (train.csv, test.csv, eval.csv)
│   ├── ft_falcon_model/          # Checkpoints y logs de Falcon
│   ├── ft_llama_model/           # Checkpoints y logs de LLaMA
│   ├── apiFalcon.ipynb           # 🚀 Script de despliegue API (Falcon)
│   ├── apiLlama.ipynb            # 🚀 Script de despliegue API (LLaMA)
│   ├── Finetuning_Falcon.ipynb   # Entrenamiento LoRA + Optuna
│   ├── Finetuning_LLaMa.ipynb    # Entrenamiento LoRA + Optuna
│   ├── MergeModels.ipynb         # Fusión de pesos (Base + LoRA)
│   └── EvalLossPlot.ipynb        # Gráficas de pérdidas (TensorBoard)
│
└── docs/                         # 📚 DOCUMENTACIÓN
    ├── memoria.pdf               # Memoria completa del TFG
    ├── PresentacionTFG.pdf       # Diapositivas de defensa
    └── PruebaChat.mp4            # Video demostrativo

---

## 🚀 Guía de Instalación y Despliegue

Debido a que los modelos LLM requieren GPU, el sistema utiliza una **arquitectura híbrida**: el modelo corre en la nube (Colab) y la aplicación en local.

### Paso 1: Desplegar la API de Inferencia (Nube)
1.  Abre `colab_notebooks/apiLlama.ipynb` (o Falcon) en Google Colab.
2.  Asegúrate de seleccionar un entorno de ejecución con **GPU (T4)**.
3.  Ejecuta todas las celdas. Esto instalará las librerías necesarias y levantará un servidor Flask con **Localtunnel**.
4.  Copia la URL pública generada al final (ej: `https://dark-pugs-sing.loca.lt`).

### Paso 2: Configurar el Backend (Local)
1.  Navega a la carpeta `chatwidget/`.
2.  Instala las dependencias de Python:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Asegúrate de tener las librerías de `DialogueKit`, `Flask` y `sqlite3` instaladas).*
3.  Abre `music_recommender.py` y actualiza la variable `API_URL` con el enlace obtenido en el Paso 1.
4.  Inicia el agente conversacional:
    ```bash
    python music_recommender.py
    ```

### Paso 3: Iniciar el Frontend (Local)
1.  En una nueva terminal, navega a la carpeta `chatwidget/`.
2.  Instala las dependencias de Node.js:
    ```bash
    npm install
    ```
3.  Lanza el servidor de desarrollo:
    ```bash
    npm start
    ```
4.  Abre tu navegador en `http://localhost:3000`. ¡El asistente está listo! 🎧

---

## 🛠️ Stack Tecnológico

*   **Lenguajes:** Python 3.10+, TypeScript.
*   **Deep Learning:** PyTorch, Transformers (Hugging Face), PEFT (LoRA).
*   **Optimización:** Optuna (Búsqueda Bayesiana de Hiperparámetros).
*   **Backend:** Flask, DialogueKit (IAI Group).
*   **Frontend:** React, WebSockets.
*   **Base de Datos:** SQLite (Catálogo de 1M+ de canciones).
*   **Infraestructura:** Google Colab (GPU T4), Localtunnel.

---

## 📝 Referencias
Este trabajo se fundamenta en la investigación de modelos generativos y su aplicación en PLN. Para más detalles técnicos, consultar la carpeta `/docs`.

*   *Maestre Gómez, A. (2025). Desarrollo de un Sistema Asistente de Música Basado en Aprendizaje Automático.* Universidad de Huelva.
*  ([https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct))
*  ([https://huggingface.co/tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b))
