# 🎵 Music Assistant based on LLMs & NLP

> **Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Informática**
> 
> *Escuela Técnica Superior de Ingeniería (ETSI), Universidad de Huelva*
>
> **Autor:** Antón Maestre Gómez | **Tutor:** Jacinto Mata Vázquez

[![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TYPESCRIPT-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/REACT-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)

[![Hugging Face](https://img.shields.io/badge/HUGGING_FACE-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/GOOGLE_COLAB-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT_/_LoRA-D00000?style=for-the-badge)](https://huggingface.co/docs/peft/index)

[![Flask](https://img.shields.io/badge/FLASK-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![DialogueKit](https://img.shields.io/badge/DIALOGUE_KIT-1155cc?style=for-the-badge)](https://github.com/iai-group/DialogueKit)

[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

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

```text
tfg_music_assistant
├── chatwidget/                   # FRONTEND AND DIALOG LOGIC
│   ├── music_recommender.py          # Main script (Chatbot orchestrator with DialogueKit)
│   └── chatwidget.md                 # Enlace al código fuente del chatwidget
│
├── colab_notebooks/              # ENTRENAMIENTO E INFERENCIA
│   ├── data/                         # Dataset (train.csv, test.csv, eval.csv)
│   ├── ft_*_model/                   # Results and final models
│   ├── api*.ipynb                    # Scripts de despliegues API 
│   ├── Finetuning_*.ipynb            # Training notebooks with LoRA and Optuna
│   ├── MergeModels.ipynb             # Script to merge LoRA weights with base model
│   └── EvalLossPlot.ipynb            # Loss plots (TensorBoard)
│
└── docs/                         # ACADEMIC DOCUMENTATION
    ├── memoria.pdf                   # Full Thesis Report (PDF)
    ├── PresentacionTFG.pdf           # Defense Presentation Slides
    └── PruebaChat.mp4                # Demo video
```
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
