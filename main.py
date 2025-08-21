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

# Neural network architecture
hidden_size = 16
output_size = 2

def relu(x):
    return np.maximum(0, x)

def policy(state, info, eval_mode=False, params=[]):
    if eval_mode:
        param_df = pd.read_json("cmaes_params.json")
        params = np.array(param_df.iloc[0]["Params"])

    state = np.array(state).flatten()
    input_size = state.shape[0]

    # Unpack parameters
    w1_end = input_size * hidden_size
    b1_end = w1_end + hidden_size
    w2_end = b1_end + hidden_size * output_size
    b2_end = w2_end + output_size

    w1 = params[:w1_end].reshape((input_size, hidden_size))
    b1 = params[w1_end:b1_end]
    w2 = params[b1_end:w2_end].reshape((hidden_size, output_size))
    b2 = params[w2_end:b2_end]

    # Forward pass
    h = relu(np.dot(state, w1) + b1)
    out = np.dot(h, w2) + b2
    action = np.tanh(out)

    return action.tolist()


def fitness(params):
    score = 0.0
    for track in range(6):  # Evaluate on all tracks for robustness
        env.unwrapped.config["track"] = track
        (obs, info) = env.reset()
        state = obs[0]
        done = False
        while not done:
            action = policy(state, info, False, params)
            (obs, _, term, trunc, info) = env.step(action)
            state = obs[0]
            done = term or trunc
        score += info["distance_covered"]
    # CMA-ES minimizes, so return negative
    return -score


def call_cma(num_gen=20, pop_size=10, num_policy_params=1):
    sigma0 = 1
    x0 = np.random.normal(0, 1, (num_policy_params, 1))
    opts = {'maxiter': num_gen, 'popsize': pop_size}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    with EvalParallel2(fitness, es.popsize + 1) as eval_all:
        while not es.stop():
            X = es.ask()
            es.tell(X, eval_all(X))
            es.logger.add()
            es.disp()
    es.result_pretty()
    return es.result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--numTracks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    train_mode = args.train
    eval_mode = args.eval
    num_tracks = args.numTracks
    seed = args.seed
    rendering = args.render

    if train_mode:
        sample_obs, _ = env.reset()
        state = sample_obs[0]
        input_size = np.array(state).flatten().shape[0]
        num_policy_params = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

        num_gen = 20
        pop_size = 10
        X = call_cma(num_gen, pop_size, num_policy_params)
        cmaes_params = X[0]
        cmaes_params_df = pd.DataFrame({'Params': [cmaes_params]})
        cmaes_params_df.to_json("cmaes_params.json")

    if rendering:
        env = RecordVideo(env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True)

    if not train_mode:
        track_score_list = []
        for t in range(num_tracks):
            env.unwrapped.config["track"] = t
            (obs, info) = env.reset(seed=seed)
            state = obs[0]
            done = False
            while not done:
                action = policy(state, info, eval_mode)
                (obs, _, term, trunc, info) = env.step(action)
                state = obs[0]
                done = term or trunc
            track_score = np.round(info["distance_covered"], 4).item()
            print("Track " + str(t) + " score:", track_score)
            track_score_list.append(track_score)

        env.close()
        perf_df = pd.DataFrame()
        perf_df["Track_number"] = [n for n in range(num_tracks)]
        perf_df["Score"] = track_score_list
        perf_df.to_json("Performance_" + str(seed) + ".json")

        plt.scatter(np.arange(len(track_score_list)), track_score_list)
        plt.xlabel("Track index")
        plt.ylabel("Scores")
        plt.title("Scores across various tracks")
        plt.savefig('Evaluation.jpg')
        plt.close()

    if train_mode:
        datContent = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]
        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []

        for i in range(1, len(datContent)):
            generations.append(int(datContent[i][0]))
            evaluations.append(int(datContent[i][1]))
            bestever.append(-float(datContent[i][4]))
            best.append(-float(datContent[i][5]))
            median.append(-float(datContent[i][6]))
            worst.append(-float(datContent[i][7]))

        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst
        logs_df.to_csv('logs.csv')

        plt.plot(generations, best, color='green')
        plt.plot(generations, median, color='blue')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.legend(["Best", "Median"])
        plt.title('Evolution of fitness across generations')
        plt.savefig('LearningCurve.jpg')
        plt.close()
