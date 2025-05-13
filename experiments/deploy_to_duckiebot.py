"""
Script para desplegar un agente entrenado en un Duckiebot real.
"""
__license__ = "MIT"

import os
import argparse
import numpy as np
import time
import logging
import cv2
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from config.config import find_and_load_config_by_seed, update_config
from duckietown_utils.env import launch_and_wrap_env

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Desactivar GPU para inferencia
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_duckiebot_image(duckiebot_ip, duckiebot_name):
    """Obtiene una imagen de la cámara del Duckiebot."""
    # Aquí implementarías la lógica para obtener la imagen del Duckiebot
    # Este es un código de ejemplo que simula la obtención de una imagen
    # En un entorno real, usarías ROS o la API de Duckietown
    
    # Simulación de obtención de imagen
    time.sleep(0.1)  # Simulando retraso de red
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # En un caso real, conectarías con la cámara del Duckiebot
    return img

def send_action_to_duckiebot(action, duckiebot_ip, duckiebot_name):
    """Envía una acción al Duckiebot."""
    # Aquí implementarías la lógica para enviar comandos al Duckiebot
    # Este es un código de ejemplo que simula el envío de comandos
    # En un entorno real, usarías ROS o la API de Duckietown
    
    left_wheel, right_wheel = action[0], action[1]
    logger.info(f"Enviando acción: izquierda={left_wheel:.2f}, derecha={right_wheel:.2f}")
    # En un caso real, enviarías estos valores al controlador del Duckiebot

def preprocess_image(img, config):
    """Preprocesa la imagen de acuerdo con la configuración del entorno."""
    # Implementar el preprocesamiento idéntico al usado durante el entrenamiento
    # Esto puede incluir recorte, redimensionamiento, normalización, etc.
    
    # Ejemplo básico de preprocesamiento
    if config["env_config"]["crop_image_top"]:
        height = img.shape[0]
        crop_height = int(height / config["env_config"]["top_crop_divider"])
        img = img[crop_height:, :, :]
    
    # Redimensionar
    input_shape = eval(config["env_config"]["resized_input_shape"])
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    
    # Convertir a escala de grises si es necesario
    if config["env_config"].get("grayscale_image", False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
    
    # Normalizar
    img = img.astype(np.float32) / 255.0
    
    # Apilar frames si es necesario
    if config["env_config"]["frame_stacking"]:
        # En un caso real, necesitarías mantener un buffer de frames anteriores
        # Por simplicidad, aquí replicamos el mismo frame
        stacked_img = np.concatenate([img] * config["env_config"]["frame_stacking_depth"], axis=2)
        return stacked_img
    
    return img

def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Ruta al checkpoint del modelo entrenado')
    parser.add_argument('--duckiebot-ip', type=str, default=os.environ.get('DUCKIEBOT_IP', '192.168.1.100'),
                        help='IP del Duckiebot')
    parser.add_argument('--duckiebot-name', type=str, default=os.environ.get('DUCKIEBOT_NAME', 'duckiebot'),
                        help='Nombre del Duckiebot')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualizar la entrada de la cámara y las acciones')
    args = parser.parse_args()

    # Inicializar Ray
    ray.init(num_cpus=1, num_gpus=0)
    register_env('Duckietown', launch_and_wrap_env)

    # Cargar configuración y modelo
    config_path = os.path.join(os.path.dirname(args.checkpoint), '..', 'params.pkl')
    if os.path.exists(config_path):
        # Cargar directamente de los parámetros guardados
        config = {}  # Cargar desde params.pkl
        # Implementar la carga desde el archivo pickle
    else:
        # Intentar encontrar por semilla
        seed = int(os.path.basename(os.path.dirname(args.checkpoint)).split('_')[1])
        config, _ = find_and_load_config_by_seed(seed)
    
    # Actualizar configuración para inferencia
    update_config(config, {'env_config': {'mode': 'inference'}})
    
    # Cargar el modelo
    trainer = PPOTrainer(config=config["rllib_config"])
    trainer.restore(args.checkpoint)
    
    print("Modelo cargado correctamente. Conectando con el Duckiebot...")
    
    # Buffer para frame stacking si es necesario
    frame_buffer = []
    
    try:
        while True:
            # Obtener imagen del Duckiebot
            img = get_duckiebot_image(args.duckiebot_ip, args.duckiebot_name)
            
            # Preprocesar imagen
            processed_img = preprocess_image(img, config)
            
            # Obtener acción del modelo
            action = trainer.compute_action(processed_img, explore=False)
            
            # Enviar acción al Duckiebot
            send_action_to_duckiebot(action, args.duckiebot_ip, args.duckiebot_name)
            
            # Visualizar si es necesario
            if args.visualize:
                cv2.imshow("Duckiebot Camera", img)
                
                # Visualizar acción (simplificado)
                action_display = np.zeros((100, 200, 3), dtype=np.uint8)
                # Dibujar representación visual de la acción
                cv2.putText(action_display, f"L: {action[0]:.2f}, R: {action[1]:.2f}", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Action", action_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)  # Controlar la frecuencia de inferencia
    
    except KeyboardInterrupt:
        print("Deteniendo el despliegue...")
    
    finally:
        # Limpiar recursos
        if args.visualize:
            cv2.destroyAllWindows()
        print("Despliegue finalizado")

if __name__ == "__main__":
    main()