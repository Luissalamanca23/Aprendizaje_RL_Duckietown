"""
Script para iniciar entrenamientos con diferentes configuraciones.
"""
__license__ = "MIT"

import os
import argparse
import random
import logging
import yaml
import subprocess
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_seed():
    """Genera un ID único para el experimento (semilla)."""
    return random.randint(1000, 9999)

def create_config_updates(args):
    """Crea un diccionario de actualizaciones de configuración basado en los argumentos."""
    config_updates = {
        "seed": args.seed if args.seed is not None else generate_seed(),
        "experiment_name": args.experiment_name,
        "env_config": {
            "domain_rand": args.domain_rand,
            "dynamics_rand": args.dynamics_rand,
            "camera_rand": args.camera_rand,
            "training_map": args.map_name,
        },
        "rllib_config": {},
        "timesteps_total": args.timesteps,
    }
    
    # Configuración del entorno
    if args.frame_stacking is not None:
        config_updates["env_config"]["frame_stacking"] = args.frame_stacking
    
    if args.frame_stacking_depth is not None:
        config_updates["env_config"]["frame_stacking_depth"] = args.frame_stacking_depth
    
    if args.grayscale is not None:
        config_updates["env_config"]["grayscale_image"] = args.grayscale
    
    if args.action_type is not None:
        config_updates["env_config"]["action_type"] = args.action_type
    
    if args.reward_function is not None:
        config_updates["env_config"]["reward_function"] = args.reward_function
    
    # Configuración de Ray/RLlib
    if args.num_workers is not None:
        config_updates["rllib_config"]["num_workers"] = args.num_workers
    
    if args.num_gpus is not None:
        config_updates["rllib_config"]["num_gpus"] = args.num_gpus
    
    if args.train_batch_size is not None:
        config_updates["rllib_config"]["train_batch_size"] = args.train_batch_size
    
    if args.lr is not None:
        config_updates["rllib_config"]["lr"] = args.lr
    
    return config_updates

def update_train_script(config_updates):
    """Actualiza el script train-rllib.py con las configuraciones proporcionadas."""
    script_path = "experiments/train-rllib.py"
    
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    # Encontrar la sección de config_updates
    config_start_idx = -1
    config_end_idx = -1
    
    for i, line in enumerate(lines):
        if "config_updates =" in line:
            config_start_idx = i
        if config_start_idx != -1 and "}" in line and config_end_idx == -1:
            config_end_idx = i
    
    if config_start_idx == -1 or config_end_idx == -1:
        logger.error("No se pudo encontrar la sección config_updates en train-rllib.py")
        return False
    
    # Generar nueva sección de configuración
    config_str = "config_updates = {\n"
    config_str += f'    "seed": {config_updates["seed"]},\n'
    config_str += f'    "experiment_name": "{config_updates["experiment_name"]}",\n'
    config_str += f'    "env_config": {{\n'
    
    for key, value in config_updates["env_config"].items():
        if isinstance(value, str):
            config_str += f'        "{key}": "{value}",\n'
        else:
            config_str += f'        "{key}": {value},\n'
    
    config_str += f'    }},\n'
    config_str += f'    "rllib_config": {{\n'
    
    for key, value in config_updates["rllib_config"].items():
        if isinstance(value, str):
            config_str += f'        "{key}": "{value}",\n'
        else:
            config_str += f'        "{key}": {value},\n'
    
    config_str += f'    }},\n'
    config_str += f'    "timesteps_total": {config_updates["timesteps_total"]},\n'
    config_str += f'}}\n'
    
    # Reemplazar la sección de configuración
    new_lines = lines[:config_start_idx] + [config_str] + lines[config_end_idx+1:]
    
    with open(script_path, 'w') as f:
        f.writelines(new_lines)
    
    return True

def run_training(args):
    """Ejecuta el entrenamiento con las configuraciones proporcionadas."""
    # Crear actualizaciones de configuración
    config_updates = create_config_updates(args)
    
    # Guardar configuración para referencia
    os.makedirs("training_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = f"training_logs/config_{config_updates['experiment_name']}_{timestamp}.yml"
    
    with open(config_file, 'w') as f:
        yaml.dump(config_updates, f, default_flow_style=False)
    
    logger.info(f"Configuración guardada en {config_file}")
    
    # Actualizar script de entrenamiento
    if args.update_script:
        if not update_train_script(config_updates):
            logger.error("No se pudo actualizar el script de entrenamiento.")
            return
    
    # Comando para ejecutar el entrenamiento
    cmd = ["python", "-m", "experiments.train-rllib"]
    
    # Configuración de GPU
    if args.gpu_id is not None:
        cmd = [f"CUDA_VISIBLE_DEVICES={args.gpu_id}"] + cmd
    
    # Configuración de CPU
    if args.cpu_list is not None:
        cmd = [f"taskset --cpu-list {args.cpu_list}"] + cmd
    
    # Usar xvfb si se especifica
    if args.use_xvfb:
        cmd = ["xvfb-run", "-a", "-s", "\"-screen 0 1400x900x24\""] + cmd
    
    # Ejecutar el comando
    logger.info(f"Iniciando entrenamiento con comando: {' '.join(cmd)}")
    
    if args.dry_run:
        logger.info("Modo dry-run. No se ejecutará el entrenamiento.")
        return
    
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error durante el entrenamiento: {e}")

def main():
    parser = argparse.ArgumentParser(description='Iniciar entrenamiento RL en Duckietown')
    
    # Parámetros generales
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Nombre descriptivo para el experimento')
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla para el experimento (se genera automáticamente si no se proporciona)')
    parser.add_argument('--timesteps', type=float, default=1.e+6,
                        help='Número total de pasos de entrenamiento')
    
    # Parámetros del entorno
    parser.add_argument('--map-name', type=str, default='multimap1',
                        help='Mapa o conjunto de mapas para entrenar')
    parser.add_argument('--domain-rand', action='store_true',
                        help='Activar randomización de dominio')
    parser.add_argument('--dynamics-rand', action='store_true',
                        help='Activar randomización de dinámica')
    parser.add_argument('--camera-rand', action='store_true',
                        help='Activar randomización de cámara')
    parser.add_argument('--frame-stacking', type=bool, default=None,
                        help='Activar apilamiento de frames')
    parser.add_argument('--frame-stacking-depth', type=int, default=None,
                        help='Número de frames a apilar')
    parser.add_argument('--grayscale', type=bool, default=None,
                        help='Convertir imágenes a escala de grises')
    parser.add_argument('--action-type', type=str, default=None,
                        choices=['heading', 'heading_smooth', 'heading_trapz', 'heading_sine', 
                                 'leftright', 'leftright_clipped', 'leftright_braking', 
                                 'steering_braking', 'discrete'],
                        help='Tipo de acción')
    parser.add_argument('--reward-function', type=str, default=None,
                        choices=['default', 'default_clipped', 'posangle', 'lane_distance'],
                        help='Función de recompensa')
    
    # Parámetros de Ray/RLlib
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Número de trabajadores para recolección de experiencias')
    parser.add_argument('--num-gpus', type=float, default=None,
                        help='Número de GPUs a utilizar')
    parser.add_argument('--train-batch-size', type=int, default=None,
                        help='Tamaño del batch de entrenamiento')
    parser.add_argument('--lr', type=float, default=None,
                        help='Tasa de aprendizaje')
    
    # Parámetros de ejecución
    parser.add_argument('--gpu-id', type=str, default=None,
                        help='ID de GPU a utilizar (ej. "0" o "0,1")')
    parser.add_argument('--cpu-list', type=str, default=None,
                        help='Lista de CPUs a utilizar (ej. "0-4" o "0,1,2")')
    parser.add_argument('--use-xvfb', action='store_true',
                        help='Usar xvfb para renderizado virtual (necesario en servidores sin pantalla)')
    parser.add_argument('--update-script', action='store_true',
                        help='Actualizar el script train-rllib.py con la configuración')
    parser.add_argument('--dry-run', action='store_true',
                        help='No ejecutar el entrenamiento, solo mostrar configuración')
    
    args = parser.parse_args()
    run_training(args)

if __name__ == '__main__':
    main()