# 🎯 Predictor de Símbolos Griegos - Arquitectura Multi-Modelo

Sistema de reconocimiento de símbolos griegos con **múltiples arquitecturas de IA** usando Flask y machine learning.

## 📋 Descripción del Proyecto

Esta aplicación web permite crear datasets, entrenar modelos de machine learning y predecir símbolos griegos (α, β, ε) dibujados por el usuario. La nueva arquitectura soporta **múltiples algoritmos de IA** que se pueden intercambiar dinámicamente con una **interfaz moderna y responsiva**.

## ⭐ Características Principales

### 🔄 **Arquitectura Multi-Modelo SOLID**
- **4 Algoritmos de IA** disponibles:
  - **Random Forest**: Robusto y eficiente para datasets pequeños
  - **Support Vector Machine (SVM)**: Excelente para patrones complejos  
  - **Red Neuronal (MLP)**: Mejor para datasets grandes y patrones complejos
  - **🆕 Red Neuronal Convolucional (CNN)**: Arquitectura profunda con TensorFlow/Keras para reconocimiento de imágenes avanzado

### 🎨 **Interfaz de Usuario Moderna**
- **Predicción en Tiempo Real**: Sin recargas de página usando AJAX
- **Canvas Interactivo**: Soporte completo para dispositivos móviles y táctiles
- **Visualización de Confianza**: Barras de progreso animadas para todas las predicciones
- **Cambio Dinámico de Modelos**: Intercambio instantáneo entre arquitecturas
- **Indicadores de Estado**: Loading states y feedback visual mejorado
- **Diseño Responsivo**: Optimizado para desktop, tablet y móvil

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

### 🧠 **Nueva Implementación CNN con TensorFlow/Keras** ✨

La versión más reciente incluye una **implementación completa de Red Neuronal Convolucional** usando TensorFlow/Keras:

#### **Características CNN:**
- **🏗️ Arquitecturas Reales**: Capas Conv2D, MaxPooling2D, BatchNormalization y Dropout
- **📐 Tres Niveles de Complejidad**:
  - **Básica**: 3 bloques convolucionales para datasets pequeños
  - **Intermedia**: 3 bloques dobles con regularización avanzada
  - **Avanzada**: 4 bloques con arquitectura profunda optimizada
- **⚡ Entrenamiento Inteligente**: 
  - Early Stopping y ReduceLROnPlateau callbacks
  - Adaptación automática según tamaño del dataset
  - Validación estratificada para datasets grandes
- **🔧 Preprocesamiento Especializado**: 
  - `CNNImagePreprocessor` optimizado para redes convolucionales
  - Normalización automática y redimensionamiento inteligente
  - Manejo de diferentes formatos de imagen (RGBA, RGB, Grayscale)

#### **Ventajas de la CNN:**
- **🎯 Mayor Precisión**: Arquitectura especializada en reconocimiento de imágenes
- **🔍 Extracción Automática**: Las capas convolucionales aprenden features automáticamente
- **📈 Escalabilidad**: Mejor rendimiento con datasets grandes
- **🔄 Transferible**: Arquitectura basada en notebook de investigación real

#### **Requisitos CNN:**
```bash
# Dependencias adicionales para CNN
pip install tensorflow>=2.10.0 keras>=2.10.0 opencv-python-headless
```

#### **Prueba Rápida CNN:**
```bash
# Ejecutar script de prueba
python test_cnn.py
```

### 🔧 **Arquitectura SOLID**

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
# Instalación básica (Random Forest, SVM, Neural Network)
pip install -r requirements.txt

# 🆕 Para usar CNN con TensorFlow/Keras (recomendado)
pip install tensorflow>=2.10.0 keras>=2.10.0 opencv-python-headless matplotlib seaborn pandas

# Verificar instalación CNN
python test_cnn.py
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
│   ├── index.html          # Página principal
│   ├── create_dataset.html # Creación de dataset
│   ├── view_dataset.html   # Visualización de dataset
│   ├── train_new.html      # ✨ Entrenamiento multi-modelo
│   └── predict_new.html    # ✨ Predicción moderna con AJAX
├── 📂 static/               # CSS, JS, imágenes
│   └── style.css           # Estilos globales
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
  - **🆕 CNN (Convolucional)**: Para máxima precisión en reconocimiento de imágenes
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

### 5. **Predecir Símbolos** ✨
- Ve a "Predecir Símbolo"
- **Selecciona el modelo** a usar en la sección superior
- **Dibuja un símbolo** en el canvas interactivo
- **Predicción instantánea**: Sin recargar la página
- **Visualiza resultados**:
  - Símbolo predicho con nivel de confianza
  - Barras de probabilidad para todas las clases
  - Información del modelo usado
  - Precisión y última fecha de entrenamiento

## 🎨 Mejoras de Interfaz de Usuario

### **Página de Predicción (predict_new.html)** ✨
- **🔄 AJAX Asíncrono**: Predicciones sin recargar la página
- **📱 Soporte Táctil**: Canvas optimizado para dispositivos móviles
- **⚡ Estados de Carga**: Indicadores visuales durante el procesamiento
- **📊 Visualización Mejorada**: 
  - Barras de progreso animadas para probabilidades
  - Tarjetas de modelo con estados visuales
  - Diseño responsive con CSS Grid
- **🎯 UX Mejorada**:
  - Feedback inmediato para acciones del usuario
  - Manejo de errores con mensajes claros
  - Transiciones suaves y animaciones

### **Características del Canvas** 🎨
```javascript
// Soporte completo para dispositivos móviles
canvas.addEventListener('touchstart', handleTouch);
canvas.addEventListener('touchmove', handleTouch);
canvas.addEventListener('touchend', handleTouch);

// Inicialización dinámica
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#800020';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}
```

### **Sistema de Barras de Probabilidad** 📊
- **Animaciones CSS**: Transiciones suaves de 0.6s
- **Datos Dinámicos**: Actualizadas via JavaScript
- **Colores Temáticos**: Gradientes que reflejan la marca
- **Responsive**: Se adaptan a diferentes tamaños de pantalla

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

### **Tecnologías Frontend** 🎨
- **HTML5**: Estructura semántica y accesible
- **CSS3**: 
  - Variables CSS customizables
  - CSS Grid y Flexbox
  - Animaciones y transiciones
  - Media queries para responsive design
- **JavaScript ES6+**:
  - Async/await para llamadas AJAX
  - Fetch API para comunicación con backend
  - Canvas API para dibujo interactivo
  - DOM manipulation moderna

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

### **Mejorar la UI** 🎨
- **Nuevos temas**: Agregar soporte para modo oscuro
- **Más visualizaciones**: Gráficos de métricas de modelo
- **Exportar resultados**: Descargar predicciones en PDF/JSON
- **Historial**: Guardar y revisar predicciones anteriores

## 🐛 Troubleshooting

### **Error: Modelo no encontrado**
- Verifica que el modelo esté entrenado
- Ve a "Entrenar Modelo" y entrena la arquitectura seleccionada

### **Error: No images available for training**
- Ve a "Crear Dataset" y dibuja símbolos primero
- Asegúrate de tener imágenes en las carpetas `dataset/alpha/`, `dataset/beta/`, `dataset/epsilon/`

### **Cambio de modelo no funciona**
- Revisa los logs en la consola del navegador
- Verifica que el modelo esté en la lista de modelos disponibles
- Revisa la consola del servidor para errores de backend

### **Canvas no responde en móvil**
- Asegúrate de que JavaScript esté habilitado
- Intenta refrescar la página
- Verifica que el navegador soporte eventos táctiles

### **Predicción tarda mucho**
- Revisa la conectividad de red
- El modelo puede estar procesando - espera a que aparezca el resultado
- Verifica los logs del servidor para errores

## 👥 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-arquitectura`)
3. Commit tus cambios (`git commit -am 'Agregar nueva arquitectura'`)
4. Push a la rama (`git push origin feature/nueva-arquitectura`)
5. Abre un Pull Request

### **Guidelines para Contribuir** 📝
- Sigue los principios SOLID
- Mantén la compatibilidad con la interfaz existente
- Agrega tests para nuevas funcionalidades
- Documenta cambios en el README
- Usa nombres descriptivos para variables y funciones

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ve el archivo `LICENSE` para detalles.

## 🎓 Contexto Académico

Proyecto desarrollado para el curso de **Gráficos por Computadora** en la **Universidad Nacional de Ingeniería (UNI)**, como ejemplo de:
- Aplicación de principios SOLID en machine learning
- Arquitecturas extensibles y mantenibles
- Integración de múltiples algoritmos de IA
- Desarrollo web moderno con Flask y Python
- **Diseño UX/UI responsivo y accesible** ✨
- **Programación frontend moderna con JavaScript ES6+** ✨

---

**Desarrollado con ❤️ por estudiantes de UNI**

## 🆕 Changelog de Versiones

### **v2.1.0** - UI Moderna y AJAX ✨
- ✅ Predicción asíncrona sin recargas de página
- ✅ Canvas optimizado para dispositivos móviles
- ✅ Barras de probabilidad animadas
- ✅ Estados de carga y mejor feedback visual
- ✅ Código JavaScript modularizado y mantenible
- ✅ Mejor manejo de errores en frontend
- ✅ Diseño responsive mejorado

### **v2.0.0** - Arquitectura Multi-Modelo
- ✅ Soporte para múltiples algoritmos de IA
- ✅ Implementación de principios SOLID
- ✅ Factory pattern para extensibilidad
- ✅ API REST para gestión de modelos
- ✅ Interfaz de cambio dinámico de modelos

### **v1.0.0** - Versión Inicial
- ✅ Funcionalidad básica con Random Forest
- ✅ Canvas de dibujo
- ✅ Sistema de dataset
- ✅ Predicción básica
