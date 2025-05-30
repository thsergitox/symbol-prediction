#!/usr/bin/env python3
"""
Script de instalación automática para CNN con TensorFlow/Keras
"""

import subprocess
import sys
import os
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Configuración automática para CNN con TensorFlow/Keras")
    print("=" * 60)
    
    # Packages required for CNN
    required_packages = [
        ("tensorflow", "tensorflow>=2.10.0"),
        ("keras", "keras>=2.10.0"),
        ("cv2", "opencv-python-headless"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"), 
        ("pandas", "pandas")
    ]
    
    # Check existing packages
    missing_packages = []
    for import_name, package_name in required_packages:
        if not check_package(import_name):
            missing_packages.append(package_name)
            print(f"❌ {import_name} no está instalado")
        else:
            print(f"✅ {import_name} ya está instalado")
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Instalando {len(missing_packages)} paquetes faltantes...")
        for package in missing_packages:
            print(f"⏳ Instalando {package}...")
            if install_package(package):
                print(f"✅ {package} instalado exitosamente")
            else:
                print(f"❌ Error instalando {package}")
                return False
    else:
        print("\n🎉 Todos los paquetes ya están instalados!")
    
    # Test the installation
    print("\n🔍 Probando la instalación...")
    try:
        # Test basic imports
        import tensorflow as tf
        from ai_architectures import ConvolutionalNeuralNetwork, CNNImagePreprocessor
        
        print(f"✅ TensorFlow {tf.__version__} importado correctamente")
        print(f"✅ GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
        print("✅ CNN classes importadas correctamente")
        
        # Test CNN initialization
        symbols_display = {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'}
        preprocessor = CNNImagePreprocessor(target_size=(64, 64))
        cnn = ConvolutionalNeuralNetwork(symbols_display, preprocessor)
        print("✅ CNN inicializada correctamente")
        
        print("\n🎉 ¡Instalación completada exitosamente!")
        print("\n📝 Próximos pasos:")
        print("1. Ejecuta: python app.py")
        print("2. Ve a: http://localhost:5000")
        print("3. Selecciona 'CNN' en la página de entrenamiento")
        print("4. ¡Disfruta de la mayor precisión en reconocimiento de símbolos!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("Ejecuta manualmente: pip install tensorflow keras opencv-python-headless")
        return False
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 