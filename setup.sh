#!/bin/bash
# Script para configurar el entorno de desarrollo

echo "Configurando entorno para Duckietown Aprendizaje RL..."

# Crear entorno virtual (opcional)
if command -v python3 -m venv &> /dev/null
then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Entorno virtual activado."
else
    echo "Advertencia: python3-venv no está instalado. Saltando creación de entorno virtual."
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

# Instalar Gym-Duckietown
echo "Instalando Gym-Duckietown..."
pip install -e git+https://github.com/duckietown/gym-duckietown.git@daffy#egg=gym-duckietown

# Configurar Weights & Biases (opcional)
echo "¿Desea configurar Weights & Biases para registrar experimentos? (s/n)"
read response
if [ "$response" = "s" ] || [ "$response" = "S" ]; then
    echo "Ingresando a Weights & Biases..."
    wandb login
fi

echo "Configuración completada. Puede iniciar el entrenamiento con:"
echo "python -m experiments.train-rllib"