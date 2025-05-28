# 🎯 Predictor de Símbolos Griegos - Arquitectura Multi-Modelo

Sistema de reconocimiento de símbolos griegos con **múltiples arquitecturas de IA** usando Flask y machine learning.

## 📋 Descripción del Proyecto

Esta aplicación web permite crear datasets, entrenar modelos de machine learning y predecir símbolos griegos (α, β, ε) dibujados por el usuario. La nueva arquitectura soporta **múltiples algoritmos de IA** que se pueden intercambiar dinámicamente.

## ⭐ Características Principales

### 🔄 **Arquitectura Multi-Modelo SOLID**
- **3 Algoritmos de IA** disponibles:
  - **Random Forest**: Robusto y eficiente para datasets pequeños
  - **Support Vector Machine (SVM)**: Excelente para patrones complejos  
  - **Red Neuronal (MLP)**: Mejor para datasets grandes y patrones complejos

### 🛠️ **Funcionalidades**
- **Crear Dataset**: Dibuja símbolos en canvas interactivo
- **Entrenar Modelos**: Entrena cualquier arquitectura con el dataset
- **Cambiar Modelos**: Intercambia entre diferentes algoritmos en tiempo real
- **Predecir Símbolos**: Usa el modelo actual para reconocer símbolos dibujados
- **Comparar Rendimiento**: Ve precisión y estadísticas de cada modelo

### 🏗️ **Arquitectura SOLID**
- ✅ **Single Responsibility**: Cada clase tiene una responsabilidad específica
- ✅ **Open/Closed**: Extensible para nuevos modelos sin modificar código existente
- ✅ **Liskov Substitution**: Modelos intercambiables através de interfaces
- ✅ **Interface Segregation**: Interfaces específicas para cada funcionalidad
- ✅ **Dependency Inversion**: Depende de abstracciones, no implementaciones

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- pip (gestor de paquetes de Python)

### 1. Clonar el repositorio
```bash
git clone https://github.com/thsergitox/symbol-prediction.git
cd symbol-prediction
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
# Usando la nueva arquitectura multi-modelo
python app_new.py

# O usando la versión original (solo Random Forest)
python app.py
```

### 5. Abrir en navegador
- Ve a: `http://localhost:5000`

## 📁 Estructura del Proyecto

```
symbol-prediction/
├── 📂 dataset/              # Imágenes de entrenamiento
│   ├── alpha/               # Imágenes de α
│   ├── beta/                # Imágenes de β
│   └── epsilon/             # Imágenes de ε
├── 📂 model/                # Modelos entrenados
│   ├── modelo_random_forest.pkl
│   ├── modelo_svm.pkl
│   └── modelo_neural_network.pkl
├── 📂 db/                   # Metadatos y configuración
│   ├── models_info.json    # Info de entrenamiento
│   └── current_model.txt   # Modelo actual seleccionado
├── 📂 templates/            # Plantillas HTML
│   ├── train_new.html      # Entrenamiento multi-modelo
│   └── predict_new.html    # Predicción multi-modelo
├── 📂 static/               # CSS, JS, imágenes
├── 📄 ai_architectures.py   # 🆕 Arquitecturas de IA (SOLID)
├── 📄 ai_manager.py         # 🆕 Manager de modelos
├── 📄 app_new.py           # 🆕 App con multi-modelo
├── 📄 ai.py                # Versión original
├── 📄 app.py               # Versión original
├── 📄 config.py            # Configuración
├── 📄 save.py              # Rutas de guardado
└── 📄 requirements.txt     # Dependencias
```

## 🎨 Guía de Uso

### 1. **Crear Dataset**
- Ve a "Crear Dataset"
- Selecciona un símbolo (α, β, ε)
- Dibuja múltiples variaciones del símbolo
- Las imágenes se guardan automáticamente

### 2. **Seleccionar Arquitectura de IA**
- Ve a "Entrenar Modelo"
- En la sección "Selección de Arquitectura de IA":
  - **Random Forest**: Para empezar rápidamente
  - **SVM**: Para mejor precisión con pocos datos
  - **Red Neuronal**: Para datasets más grandes
- Haz clic en "Cambiar a [Modelo]"

### 3. **Entrenar Modelo**
- Asegúrate de tener imágenes en el dataset
- Haz clic en "Entrenar Modelo [Tipo]"
- El sistema mostrará la precisión obtenida
- Cada modelo se guarda independientemente

### 4. **Comparar Modelos**
- En la página de entrenamiento verás todas las arquitecturas
- Cada tarjeta muestra:
  - Estado (Entrenado/No Entrenado)
  - Precisión del modelo
  - Descripción del algoritmo

### 5. **Predecir Símbolos**
- Ve a "Predecir Símbolo"
- Dibuja un símbolo en el canvas
- Haz clic en "Predecir con [Modelo Actual]"
- Ve la predicción y nivel de confianza

## 🔧 API REST (Opcional)

La nueva arquitectura incluye endpoints API:

```bash
# Obtener información de todos los modelos
GET /api/models

# Cambiar modelo actual
POST /api/models/switch
{
  "model_type": "svm"
}

# Obtener info del modelo actual
GET /api/models/current

# Obtener info de modelo específico
GET /api/models/{model_type}/info
```

## 📊 Características Técnicas

### **Modelos Disponibles**
| Modelo | Tipo | Ventajas | Ideal Para |
|--------|------|----------|------------|
| Random Forest | Ensemble | Rápido, robusto | Datasets pequeños |
| SVM | Kernel RBF | Alta precisión | Patrones complejos |
| Red Neuronal | MLP | Aprende patrones complejos | Datasets grandes |

### **Parámetros por Defecto**
- **Tamaño de imagen**: 100x100 píxeles
- **Formato**: Escala de grises
- **Intervalo de entrenamiento**: 5 minutos mínimo
- **Split de datos**: 80% entrenamiento, 20% prueba

## 🎯 Principios SOLID Implementados

### **1. Single Responsibility Principle (SRP)**
- `ModelInterface`: Solo define interfaz de modelos
- `DataPreprocessorInterface`: Solo preprocesamiento  
- `AIService`: Solo orquestación de IA
- `ModelManager`: Solo gestión de modelos

### **2. Open/Closed Principle (OCP)**
```python
# ✅ Extensible - Agregar nuevo modelo sin modificar código existente
class DeepLearningModel(ModelInterface):
    def train(self, X, y): ...
    def predict(self, X): ...
    # ...

# Registrar nuevo modelo
ModelFactory.register_model('deep_learning', DeepLearningModel)
```

### **3. Liskov Substitution Principle (LSP)**
```python
# ✅ Cualquier modelo puede sustituir a otro
model: ModelInterface = ModelFactory.create_model('svm', ...)
model = ModelFactory.create_model('neural_network', ...)  # Intercambiable
```

### **4. Interface Segregation Principle (ISP)**
- Interfaces específicas y pequeñas
- `ModelInterface` para modelos
- `DataPreprocessorInterface` para preprocesamiento

### **5. Dependency Inversion Principle (DIP)**
```python
# ✅ Depende de abstracciones, no implementaciones concretas
class AIService:
    def __init__(self, model: ModelInterface):  # Interfaz, no clase concreta
        self.model = model
```

## 🔮 Extensibilidad Futura

### **Agregar Nuevos Modelos**
```python
# 1. Crear nueva clase que implemente ModelInterface
class TransformerModel(ModelInterface):
    def train(self, X, y): ...
    def predict(self, X): ...
    # ...

# 2. Registrar en el factory
ModelFactory.register_model('transformer', TransformerModel)

# 3. ¡Listo! Aparecerá automáticamente en la UI
```

### **Agregar Nuevos Preprocesadores**
```python
class AdvancedPreprocessor(DataPreprocessorInterface):
    def preprocess_image(self, image): ...
    # Implementar preprocesamiento avanzado
```

## 🐛 Troubleshooting

### **Error: Modelo no encontrado**
- Verifica que el modelo esté entrenado
- Ve a "Entrenar Modelo" y entrena la arquitectura seleccionada

### **Error: No images available for training**
- Ve a "Crear Dataset" y dibuja símbolos primero
- Asegúrate de tener imágenes en las carpetas `dataset/alpha/`, `dataset/beta/`, `dataset/epsilon/`

### **Cambio de modelo no funciona**
- Revisa los logs en la consola
- Verifica que el modelo esté en la lista de modelos disponibles

## 👥 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-arquitectura`)
3. Commit tus cambios (`git commit -am 'Agregar nueva arquitectura'`)
4. Push a la rama (`git push origin feature/nueva-arquitectura`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ve el archivo `LICENSE` para detalles.

## 🎓 Contexto Académico

Proyecto desarrollado para el curso de **Gráficos por Computadora** en la **Universidad Nacional de Ingeniería (UNI)**, como ejemplo de:
- Aplicación de principios SOLID en machine learning
- Arquitecturas extensibles y mantenibles
- Integración de múltiples algoritmos de IA
- Desarrollo web con Flask y Python

---

**Desarrollado con ❤️ por estudiantes de UNI**
