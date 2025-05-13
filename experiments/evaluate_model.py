"""
Script para evaluar un modelo entrenado con diferentes mapas y condiciones.
"""
__license__ = "MIT"

import os
import argparse
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import matplotlib.pyplot as plt

from config.config import find_and_load_config_by_seed, update_config
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.trajectory_plot import correct_gym_duckietown_coordinates, plot_trajectories

def evaluate_model(args):
    # Inicializar Ray
    ray.init(num_cpus=1, num_gpus=0)
    register_env('Duckietown', launch_and_wrap_env)
    
    # Cargar configuración y modelo
    if args.seed_model_id:
        config, checkpoint_path = find_and_load_config_by_seed(
            args.seed_model_id, 
            preselected_experiment_idx=args.experiment_idx,
            preselected_checkpoint_idx=args.checkpoint_idx
        )
    else:
        # Cargar directamente desde el checkpoint
        checkpoint_path = args.checkpoint
        config_path = os.path.join(os.path.dirname(checkpoint_path), '..', 'params.pkl')
        if not os.path.exists(config_path):
            raise ValueError(f"No se encontró el archivo de configuración en {config_path}")
        
        # Implementar carga desde archivo pickle
        # config = ...
    
    # Actualizar configuración para evaluación
    update_config(config, {
        'env_config': {
            'mode': 'inference',
            'training_map': args.map_name,
            'domain_rand': args.domain_rand,
            'dynamics_rand': args.dynamics_rand,
            'camera_rand': args.camera_rand
        }
    })
    
    # Cargar el modelo
    trainer = PPOTrainer(config=config["rllib_config"])
    trainer.restore(checkpoint_path)
    
    print(f"Modelo cargado desde {checkpoint_path}")
    print(f"Evaluando en mapa: {args.map_name}")
    
    # Preparar directorio de resultados
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Recolectar métricas para cada episodio
    episode_rewards = []
    distances_traveled = []
    timeouts = []
    crashes = []
    trajectories = []
    
    # Evaluar el modelo en múltiples episodios
    for i in range(args.num_episodes):
        print(f"Evaluando episodio {i+1}/{args.num_episodes}...")
        
        # Crear entorno
        env = launch_and_wrap_env(config["env_config"])
        
        # Ejecutar episodio
        obs = env.reset()
        done = False
        total_reward = 0
        trajectory = []
        
        while not done:
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Guardar posición para trayectoria
            trajectory.append(correct_gym_duckietown_coordinates(env.unwrapped, env.unwrapped.cur_pos))
            
            # Renderizar si es necesario
            if args.render:
                env.render('human' if not args.top_view else 'top_down')
        
        # Guardar métricas del episodio
        episode_rewards.append(total_reward)
        trajectories.append(trajectory)
        
        # Determinar motivo del final del episodio
        if 'Simulator' in info and 'msg' in info['Simulator']:
            if 'timeout' in info['Simulator']['msg']:
                timeouts.append(1)
                crashes.append(0)
            elif 'crash' in info['Simulator']['msg']:
                timeouts.append(0)
                crashes.append(1)
            else:
                timeouts.append(0)
                crashes.append(0)
        else:
            timeouts.append(0)
            crashes.append(0)
        
        # Calcular distancia recorrida (aproximada)
        distance = 0
        for i in range(1, len(trajectory)):
            distance += np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
        distances_traveled.append(distance)
        
        # Cerrar entorno
        env.close()
    
    # Calcular estadísticas
    avg_reward = np.mean(episode_rewards)
    avg_distance = np.mean(distances_traveled)
    crash_rate = sum(crashes) / args.num_episodes
    timeout_rate = sum(timeouts) / args.num_episodes
    
    # Imprimir resultados
    print("\n=== Resultados de Evaluación ===")
    print(f"Recompensa promedio: {avg_reward:.2f}")
    print(f"Distancia promedio recorrida: {avg_distance:.2f}")
    print(f"Tasa de choques: {crash_rate:.2f}")
    print(f"Tasa de timeouts: {timeout_rate:.2f}")
    
    # Guardar resultados
    results = {
        'rewards': episode_rewards,
        'distances': distances_traveled,
        'crashes': crashes,
        'timeouts': timeouts,
        'avg_reward': avg_reward,
        'avg_distance': avg_distance,
        'crash_rate': crash_rate,
        'timeout_rate': timeout_rate
    }
    
    # Guardar resultados en formato texto
    results_file = os.path.join(args.results_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("=== Resultados de Evaluación ===\n")
        f.write(f"Modelo: {checkpoint_path}\n")
        f.write(f"Mapa: {args.map_name}\n")
        f.write(f"Episodios: {args.num_episodes}\n\n")
        f.write(f"Recompensa promedio: {avg_reward:.2f}\n")
        f.write(f"Distancia promedio recorrida: {avg_distance:.2f}\n")
        f.write(f"Tasa de choques: {crash_rate:.2f}\n")
        f.write(f"Tasa de timeouts: {timeout_rate:.2f}\n")
    
    # Generar visualizaciones de trayectorias
    if args.plot_trajectories:
        # Graficar todas las trayectorias
        plt.figure(figsize=(10, 10))
        plot_trajectories(trajectories, show_plot=False)
        plt.title(f"Trayectorias en {args.map_name}")
        plt.savefig(os.path.join(args.results_dir, 'trajectories.png'))
        
        # Graficar trayectorias con punto de inicio unificado
        plt.figure(figsize=(10, 10))
        plot_trajectories(trajectories, show_plot=False, unify_start_tile=True)
        plt.title(f"Trayectorias (inicio unificado) en {args.map_name}")
        plt.savefig(os.path.join(args.results_dir, 'trajectories_unified.png'))
    
    print(f"Resultados guardados en {args.results_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluar un modelo entrenado')
    
    # Grupo de argumentos para seleccionar el modelo
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--seed-model-id', type=int, 
                             help='ID del modelo a evaluar (formato de 4 dígitos, ej. 3045)')
    model_group.add_argument('--checkpoint', type=str,
                             help='Ruta directa al checkpoint del modelo')
    
    # Argumentos para find_and_load_config_by_seed
    parser.add_argument('--experiment-idx', type=int, default=0,
                        help='Índice del experimento si hay múltiples con la misma semilla')
    parser.add_argument('--checkpoint-idx', type=int, default=0,
                        help='Índice del checkpoint (0: mejor, 1: final)')
    
    # Argumentos para la evaluación
    parser.add_argument('--map-name', default='loop_empty',
                        help='Mapa para evaluación')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Número de episodios para evaluación')
    parser.add_argument('--results-dir', default='./evaluation_results',
                        help='Directorio para guardar resultados')
    parser.add_argument('--domain-rand', action='store_true',
                        help='Activar randomización de dominio')
    parser.add_argument('--dynamics-rand', action='store_true',
                        help='Activar randomización de dinámica')
    parser.add_argument('--camera-rand', action='store_true',
                        help='Activar randomización de cámara')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar durante la evaluación')
    parser.add_argument('--top-view', action='store_true',
                        help='Usar vista superior durante la renderización')
    parser.add_argument('--plot-trajectories', action='store_true',
                        help='Generar gráficos de trayectorias')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == '__main__':
    main()