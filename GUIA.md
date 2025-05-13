# Guía Completa: Duckietown Aprendizaje por Refuerzo

Esta guía proporciona instrucciones detalladas para:
1. Configurar el entorno
2. Entrenar un agente de seguimiento de carril
3. Evaluar el rendimiento del agente
4. Desplegar el agente en un Duckiebot real

## 1. Configuración del Entorno

### Requisitos previos
- Python 3.7+
- Pip
- Git
- (Opcional) GPU con CUDA para entrenamiento acelerado

### Pasos de instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu_usuario/duckietown-aprendizaje-rl.git
   cd duckietown-aprendizaje-rl
   ```

2. **Utilizar el script de configuración automática:**
   ```bash
   ./setup.sh
   ```
   
   Este script:
   - Crea un entorno virtual (opcional)
   - Instala dependencias desde requirements.txt
   - Instala Gym-Duckietown
   - Configura Weights & Biases (opcional)

3. **Instalación manual (alternativa):**
   ```bash
   # Crear y activar entorno virtual (opcional)
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   
   # Instalar Gym-Duckietown
   pip install -e git+https://github.com/duckietown/gym-duckietown.git@daffy#egg=gym-duckietown
   
   # Configurar Weights & Biases
   wandb login
   ```

## 2. Entrenamiento del Agente

Existen dos métodos para iniciar el entrenamiento:

### Método Básico
```bash
python -m experiments.train-rllib
```

### Método Avanzado (Recomendado)
Utiliza el script `train_model.py` para mayor flexibilidad:

```bash
# Ejemplo básico:
python -m experiments.train_model --experiment-name "MiEntrenamiento" --domain-rand --map-name "multimap1"

# Ejemplo completo:
python -m experiments.train_model \
    --experiment-name "EntrenamientoCompleto" \
    --domain-rand \
    --dynamics-rand \
    --camera-rand \
    --map-name "multimap1" \
    --frame-stacking True \
    --frame-stacking-depth 3 \
    --action-type "heading" \
    --reward-function "posangle" \
    --num-workers 8 \
    --num-gpus 1 \
    --train-batch-size 4096 \
    --lr 5e-5 \
    --timesteps 1e6 \
    --gpu-id 0 \
    --use-xvfb
```

#### Parámetros clave para el entrenamiento:

| Parámetro | Descripción | Valor recomendado |
|-----------|-------------|-------------------|
| `--experiment-name` | Nombre descriptivo del experimento | Un nombre significativo |
| `--domain-rand` | Activar randomización del dominio | Activado para sim-to-real |
| `--dynamics-rand` | Activar randomización de la dinámica | Activado para sim-to-real |
| `--camera-rand` | Activar randomización de la cámara | Activado para sim-to-real |
| `--map-name` | Mapa o conjunto de mapas | "multimap1" |
| `--action-type` | Tipo de acción | "heading" |
| `--reward-function` | Función de recompensa | "posangle" |
| `--num-workers` | Número de trabajadores en paralelo | 8-16 (según CPU) |
| `--num-gpus` | Número de GPUs a utilizar | 1 |
| `--timesteps` | Total de pasos de entrenamiento | 1e6 - 5e6 |

### Monitorización del Entrenamiento

Durante el entrenamiento, puedes monitorizar el progreso en Weights & Biases:
1. Abre tu navegador y ve a [wandb.ai](https://wandb.ai)
2. Inicia sesión y ve a tu proyecto
3. Observa métricas como recompensa, longitud de episodio, pérdidas, etc.

## 3. Evaluación del Modelo

Una vez entrenado el modelo, evalúa su rendimiento:

### Evaluación Básica
```bash
python -m experiments.test-rllib
```

### Evaluación Avanzada (Recomendado)
Utiliza el script `evaluate_model.py` para análisis detallados:

```bash
# Evaluar por ID de semilla:
python -m experiments.evaluate_model \
    --seed-model-id 1234 \
    --map-name "loop_empty" \
    --num-episodes 10 \
    --results-dir "./resultados_evaluacion" \
    --plot-trajectories \
    --render

# Evaluar por checkpoint específico:
python -m experiments.evaluate_model \
    --checkpoint "./artifacts/PPO_MiEntrenamiento/checkpoint_000010/checkpoint-10" \
    --map-name "loop_empty" \
    --num-episodes 10 \
    --results-dir "./resultados_evaluacion" \
    --plot-trajectories
```

### Visualización de Trayectorias
```bash
python -m experiments.test-rllib --analyse-trajectories --results-path MisResultados
```

### Vista Superior (Bird's Eye)
```bash
python -m experiments.test-rllib --top-view
```

## 4. Despliegue en un Duckiebot Real

### Preparación del Duckiebot
1. Asegúrate de que tu Duckiebot esté encendido y conectado a la misma red
2. Verifica que la cámara esté funcionando correctamente
3. Configura las variables de entorno:
   ```bash
   export DUCKIEBOT_IP=<IP_DEL_DUCKIEBOT>
   export DUCKIEBOT_NAME=<NOMBRE_DEL_DUCKIEBOT>
   ```

### Despliegue del Modelo
```bash
# Método básico:
python -m experiments.deploy_to_duckiebot \
    --checkpoint "./artifacts/PPO_MiEntrenamiento/checkpoint_000010/checkpoint-10" \
    --visualize

# Con IP específica:
python -m experiments.deploy_to_duckiebot \
    --checkpoint "./artifacts/PPO_MiEntrenamiento/checkpoint_000010/checkpoint-10" \
    --duckiebot-ip 192.168.1.100 \
    --duckiebot-name duckiebot \
    --visualize
```

## 5. Configuración Avanzada

### Configuración del Entorno
Puedes modificar parámetros avanzados en:
- `config/config.yml`: Configuración general del entorno
- `config/algo/ppo.yml`: Configuración específica del algoritmo PPO

### Randomización de Dominio
La randomización de dominio es crucial para la transferencia sim-to-real:

```bash
python -m experiments.train_model \
    --experiment-name "SimToReal" \
    --domain-rand \
    --dynamics-rand \
    --camera-rand \
    --timesteps 2e6
```

## 6. Solución de Problemas

### Problemas comunes:

1. **Error: "Could not connect to display"**
   - Solución: Utiliza xvfb para renderizado sin pantalla
   ```bash
   xvfb-run -a -s "-screen 0 1400x900x24" python -m experiments.train-rllib
   ```
   - O usa la opción `--use-xvfb` en `train_model.py`

2. **Memoria insuficiente**
   - Solución: Reduce `train_batch_size` y `num_workers`
   ```bash
   python -m experiments.train_model --train-batch-size 2048 --num-workers 4
   ```

3. **Entrenamiento lento**
   - Solución: Usa GPU y ajusta número de trabajadores
   ```bash
   python -m experiments.train_model --num-gpus 1 --num-workers 16
   ```

4. **Problemas con Weights & Biases**
   - Solución: Reinicia la sesión
   ```bash
   wandb login --relogin
   ```

## 7. Ejemplos Completos

### Entrenamiento completo con randomización
```bash
python -m experiments.train_model \
    --experiment-name "SimToReal_Complete" \
    --domain-rand \
    --dynamics-rand \
    --camera-rand \
    --map-name "multimap1" \
    --action-type "heading" \
    --reward-function "posangle" \
    --num-workers 16 \
    --num-gpus 1 \
    --train-batch-size 4096 \
    --lr 5e-5 \
    --timesteps 2e6 \
    --use-xvfb
```

### Evaluación exhaustiva
```bash
python -m experiments.evaluate_model \
    --seed-model-id 1234 \
    --map-name "loop_empty" \
    --num-episodes 20 \
    --results-dir "./evaluacion_detallada" \
    --plot-trajectories \
    --domain-rand \
    --dynamics-rand \
    --camera-rand
```

### Despliegue con visualización
```bash
python -m experiments.deploy_to_duckiebot \
    --checkpoint "./artifacts/PPO_SimToReal_Complete/checkpoint_000020/checkpoint-20" \
    --duckiebot-ip 192.168.1.100 \
    --visualize
```