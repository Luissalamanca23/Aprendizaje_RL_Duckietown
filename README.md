# Duckietown Aprendizaje por Refuerzo

Este repositorio contiene código para entrenar, evaluar y desplegar un agente de aprendizaje por refuerzo en un Duckiebot.

## Configuración

### Requisitos

- Python 3.7+
- [Gym-Duckietown](https://github.com/duckietown/gym-duckietown) 
- Otras dependencias en `requirements.txt`

### Instalación

1. Clonar este repositorio:
```bash
git clone https://github.com/tu_usuario/duckietown-aprendizaje-rl.git
cd duckietown-aprendizaje-rl
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Instalar Gym-Duckietown:
```bash
pip install -e git+https://github.com/duckietown/gym-duckietown.git@daffy#egg=gym-duckietown
```

## Flujo de trabajo

### Entrenar un agente

Para entrenar un agente de seguimiento de carril:

```bash
python -m experiments.train-rllib
```

Para modificar los parámetros de entrenamiento, edite el archivo `experiments/train-rllib.py` y cambie las líneas 31-35:

```python
config_updates = {"seed": 1234,  # Identificador único del experimento
                  "experiment_name": "MiEntrenamiento",
                  "env_config": {"domain_rand": True,  # Randomización de dominio
                                 "dynamics_rand": True,  # Randomización de dinámica
                                 "camera_rand": True},  # Randomización de cámara
                  "rllib_config": {},
                  "timesteps_total": 1.e+6,  # Total de pasos de entrenamiento
                  }
```

### Evaluar un agente entrenado

Para probar el agente en simulación:

```bash
python -m experiments.test-rllib
```

Para evaluar el rendimiento y visualizar trayectorias:

```bash
python -m experiments.test-rllib --analyse-trajectories --results-path EvaluationResults
```

Para visualizar desde una vista superior:

```bash
python -m experiments.test-rllib --top-view
```

### Desplegar en el Duckiebot

Para desplegar el modelo entrenado en un Duckiebot real:

1. Asegúrese de que su Duckiebot esté encendido y conectado a la misma red que su computadora.

2. Configure las variables de entorno con la dirección IP de su Duckiebot:
```bash
export DUCKIEBOT_IP=<IP_DE_TU_DUCKIEBOT>
export DUCKIEBOT_NAME=<NOMBRE_DE_TU_DUCKIEBOT>
```

3. Ejecute el script de despliegue:
```bash
python -m experiments.deploy_to_duckiebot --checkpoint ./artifacts/PPO_<NOMBRE_DEL_EXPERIMENTO>/checkpoint_<NÚMERO>/checkpoint-<NÚMERO>
```

## Monitoreo con Weights & Biases

El entrenamiento utiliza Weights & Biases para monitorizar el progreso. Antes de entrenar, inicie sesión en su cuenta:

```bash
wandb login
```

## Configuración Avanzada

Para opciones de configuración adicionales, consulte los archivos en:
- `config/config.yml`: Configuración general del entorno
- `config/algo/ppo.yml`: Configuración específica del algoritmo PPO

## Resolución de problemas

Si encuentra problemas con la visualización, asegúrese de tener X11 configurado correctamente (para usuarios de macOS y WSL).