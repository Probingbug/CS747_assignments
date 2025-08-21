import highway_env
import gymnasium
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
import argparse
import warnings
warnings.filterwarnings('ignore')

env = gymnasium.make('racetrack-v0', render_mode='rgb_array')

def policy(state, info, eval_mode = False, params = []):

    if eval_mode:
        param_df = pd.read_json("cmaes_params.json")
        params = np.array(param_df.iloc[0]["Params"])

    obs = state[0]  # 13x13 binary matrix
    car_row, car_col = 12, 6  # Car position in matrix

    # Find all lane center positions
    lane_positions = np.argwhere(obs == 1)

    if len(lane_positions) == 0:
        # No visible lanes, slow down
        return [-1.0, 0.0]

    # Choose the closest lane center
    distances = np.linalg.norm(lane_positions - np.array([car_row, car_col]), axis=1)
    target = lane_positions[np.argmin(distances)]

    # Determine steering direction
    steering = (target[1] - car_col) / 6.0  # Normalize to roughly [-1, 1]
    steering = np.clip(steering, -1, 1)

    # Speed control based on curvature (more centers = straighter path)
    num_centers_ahead = np.sum(obs[0:7, :])
    acceleration = 0.5 if num_centers_ahead > 3 else 0.0

    return [acceleration, steering]

def fitness(params):
    fitness_value = 0.0
    test_tracks = [0, 1, 2]
    for track in test_tracks:
        env.unwrapped.config["track"] = track
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy(obs, info, False, params)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            episode_reward += info.get("distance_covered", 0)
        fitness_value += episode_reward
    return -fitness_value  # CMA-ES minimizes by default

def call_cma(num_gen=2, pop_size=2, num_policy_params = 1):
  sigma0 = 1
  x0 = np.random.normal(0, 1, (num_policy_params, 1))  # Initialisation of parameter vector
  opts = {'maxiter':num_gen, 'popsize':pop_size}
  es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
  with EvalParallel2(fitness, es.popsize + 1) as eval_all:
    while not es.stop():
      X = es.ask()
      es.tell(X, eval_all(X))
      es.logger.add()  # write data to disc for plotting
      es.disp()
  es.result_pretty()
  return es.result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--numTracks', type=int, default=6)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.train:
        result = call_cma(num_gen=10, pop_size=6, num_policy_params=1)
        result_dict = {"Params": result[0].tolist()}
        pd.DataFrame([result_dict]).to_json("cmaes_params.json")

    else:
        track_indices = list(range(args.numTracks))
        for track in track_indices:
            env.unwrapped.config["track"] = track
            obs, info = env.reset()
            if args.render:
                env = RecordVideo(env, video_folder="videos", name_prefix=f"track{track}_seed{args.seed}")
            done = False
            while not done:
                action = policy(obs, info, eval_mode=args.eval)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            print(f"Track {track} | Distance Covered: {info['distance_covered']} | On Road: {info['on_road']}")