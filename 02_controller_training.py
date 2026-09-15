"""PPO training entry point for behaviour-aware setpoint control.

``--ob_model_causality`` selects which occupant drives the simulation. ``0``
(non-causal) and ``4`` (causal) are the two arms compared in the paper: both
load a data-driven checkpoint, which then replaces the rule-based occupant and
decides each override itself, and both raise a clear error when the checkpoint
is absent. ``-1`` runs the rule-based occupant and needs no checkpoint.
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import torch

from multiprocessing import freeze_support
import os
import numpy as np
import argparse
import pandas as pd
import shutil
import sys
import matplotlib.pyplot as plt
import gc
from itertools import product

from rl_env import EnergyPlusEnv

sys.setrecursionlimit(10000000)

parser = argparse.ArgumentParser(description='RL parameters')
parser.add_argument('--start_month', type=int, default=6)
parser.add_argument('--start_day', type=int, default=1)
parser.add_argument('--end_month', type=int, default=8)
parser.add_argument('--end_day', type=int, default=31)

parser.add_argument('--rl_option', type=int, default=1)
parser.add_argument('--occ_option', type=int, default=0)
parser.add_argument('--save_csv', type=int, default=0)
parser.add_argument('--save_trajectory', type=int, default=0)
parser.add_argument('--only_last_trajectory', type = int, default = 0)
parser.add_argument('--model_name', type=str, default="model")

parser.add_argument('--clean_scenarios', type=int, default=1)
parser.add_argument('--print_values', type=int, default=0)
parser.add_argument('--reset_previous_training', type=int, default=0)

parser.add_argument('--log_tensorboard', type = int, default = 0)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--learning_rate', type = float, default = 0.00001)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--n_epochs', type = int, default = 10)
parser.add_argument('--model_verbose', type = int, default = 0)
parser.add_argument('--n_nodes', type = int, default = 64)
parser.add_argument('--n_layers', type = int, default = 2)
parser.add_argument('--learning_days', type = int, default = 3)
parser.add_argument("--obj_coeffs", nargs='+', type=float, default = [1, 1])
parser.add_argument('--reward_form', type = str, default = "positive")
parser.add_argument('--occ_case', type = int, default = 0)
parser.add_argument('--rl_occ_obs', type=int, default=1)
parser.add_argument('--ob_model_causality', type = int, default = 4,
                    help = "occupant driving the simulation: 0 non-causal model, 4 causal model, -1 rule-based")

parser.add_argument('--num_episode', type = int, default = 4)
parser.add_argument('--n_envs', type = int, default = 8)

args=parser.parse_args()

sim_minutes = 15
sim_steps_per_day = int(24*60/sim_minutes)

def main():


    c1_n_nodes = [64]
    c2_n_layers = [4]
    c3_n_epochs = [10]
    c4_batch_size = [128]

    c5_learning_rate = [0.00005]
    c6_discount_factor = [0.99]
    c7_learning_days = [4]


    c8_occ_option = [1]
    c11_obj_coeffs = [[1,0]] # This can be adjusted for different reward formulation.
    c12_reward_form = ["positive"]
    c13_occ_case = [0]

    # --ob_model_causality picks the checkpoint that replaces the rule-based
    # occupant. 0 and 4 are
    # the two arms compared in the paper; -1 trains without an occupant model.
    OB_MODELS = {
        -1: None,
        0: "non_causal_Jun-Aug_26_GridSearch_occ{occ}",                 # non-causal
        4: "causal_manual_onlyR_lastR_Jun-Aug_26_GridSearch_occ{occ}",  # causal
    }
    if args.ob_model_causality not in OB_MODELS:
        raise ValueError(
            f"--ob_model_causality must be one of {sorted(OB_MODELS)}, got {args.ob_model_causality}"
        )

    name_template = OB_MODELS[args.ob_model_causality]
    c15_ob_model = name_template.format(occ=c13_occ_case[0]) if name_template else None
    model_path = f"occupant_models/{c15_ob_model}.pth" if c15_ob_model else None
    c15_ob_model_int = [args.ob_model_causality if name_template else None]

    c14_rl_occ_obs = c8_occ_option
    c16_ent_coef = [0]
    c17_clip_range = [0.2]
    c18_vf_coef = [0.5]
    c19_gae_lambda = [0.95]
    c20_max_grad_norm = [0.5]


    if c15_ob_model is not None:
        from occ_model_collections import LSTMOrdinalLogisticRegression

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Occupant-model checkpoint not found: {model_path}. "
                "Use --ob_model_causality -1 to train without an occupant model."
            )
        model_data = torch.load(model_path, weights_only = False)

        ob_model = LSTMOrdinalLogisticRegression(input_size=model_data["config"]["input_size"],
                            hidden_size=model_data["config"]["hidden_size"],
                            num_layers=model_data["config"]["num_layers"],
                            num_classes=model_data["config"]["num_classes"], 
                            specific_label=model_data["config"]["specific_label"],
                            dropout_ratio=model_data["config"]["dropout_ratio"],
                            last_step_direct=model_data["config"].get("last_step_direct", False))
        
        ob_model.load_state_dict(model_data["model_state_dict"])
        
        print("\n", c15_ob_model)
        print(ob_model.eval())
        scaler_train = model_data["scaler"]
        
    else: 
        ob_model = None
        scaler_train = None

    experimental_cases = list(product(c1_n_nodes, c2_n_layers, c3_n_epochs, c4_batch_size, 
                c5_learning_rate, c6_discount_factor, c7_learning_days, c8_occ_option,
                c11_obj_coeffs, c12_reward_form, c13_occ_case, c14_rl_occ_obs, c15_ob_model_int,
                c16_ent_coef, c17_clip_range, c18_vf_coef, c19_gae_lambda, c20_max_grad_norm))

    case_params = experimental_cases[0]

    (args.n_nodes, args.n_layers, args.n_epochs, args.batch_size, args.learning_rate, args.discount_factor, 
    args.n_learning_days, args.occ_option, args.obj_coeffs, args.reward_form, args.occ_case, 
    args.rl_occ_obs, args.ob_model_causality, ent_coef, clip_range, vf_coef, gae_lambda, max_grad_norm) = case_params


    args.model_name = f"w_{args.obj_coeffs[0]}_{args.obj_coeffs[1]}_" \
                     f"OBM-{c15_ob_model}_nn_{args.n_nodes}_nl_{args.n_layers}_lr_{args.learning_rate}_" \
                     f"oc_{args.occ_option}_ot_{args.occ_case}_{args.reward_form}_ld_{args.n_learning_days}_ne_{args.n_epochs}_bs_{args.batch_size}_" \
                     f"dc_{args.discount_factor}_et_{ent_coef}_cr_{clip_range}_vf_{vf_coef}_gae_{gae_lambda}_gn_{max_grad_norm}"

    print(args.model_name)
    print(args.occ_option)

    args.obj_coeffs = args.obj_coeffs/np.sum(args.obj_coeffs)

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"    
    
    print("# CPU : ", torch.get_num_threads())
    
    gc.collect()
    gc.enable()
    freeze_support()

    print("\n--new training--")
    progress = 0

    print("\n",progress," - Check if we need to reset the previous training"); progress += 1
    model_name = args.model_name

    print("model_name = ", model_name)

    if args.reset_previous_training == 1:
        if os.path.exists("rl_results/{model_name}/{model_name}.zip".format(model_name = args.model_name)):
            os.remove("rl_results/{model_name}/{model_name}.zip".format(model_name = args.model_name))

        if os.path.exists("rl_results/{model_name}".format(model_name = args.model_name)):
            shutil.rmtree("rl_results/{model_name}".format(model_name = args.model_name))

    model_folder = "rl_results/{}".format(model_name)
    os.makedirs(model_folder, exist_ok=True)

    scenarios_folder = os.path.join(model_folder, "scenarios")
    os.makedirs(scenarios_folder, exist_ok=True)


    log_dir = os.path.join(model_folder, "monitor_logs")
    os.makedirs(log_dir, exist_ok=True)

    print("\n",progress," - Environment is being created"); progress += 1

    env_config=dict(
    csv= False,
    verbose= False,
    output="rl_results/{model_name}/scenarios".format(model_name = args.model_name))

    env_kwargs = dict(env_config = env_config, 
                    start_month=args.start_month, start_day=args.start_day, 
                    end_month=args.end_month, end_day=args.end_day,
                    rl_option = args.rl_option, occ_option = args.occ_option,
                    save_csv = args.save_csv,
                    print_values = args.print_values, model_name = args.model_name,
                    obj_coeffs = args.obj_coeffs,
                    reward_form= args.reward_form, occ_case= args.occ_case,
                    rl_occ_obs = args.rl_occ_obs,
                    dd_ob_model = ob_model, scaler = scaler_train,
                    baseline_temp = 26, ob_causal = args.ob_model_causality)
    
    if os.path.exists("rl_results/{model_name}/monitor_logs/0.monitor.csv".format(model_name = args.model_name)):
        monitor_kwargs = dict(override_existing = False)
    else:
        monitor_kwargs = dict(override_existing = True)

    n_envs = args.n_envs

    while True:
        try:
            env = make_vec_env(EnergyPlusEnv, n_envs = n_envs, monitor_dir=log_dir, monitor_kwargs = monitor_kwargs, env_kwargs = env_kwargs, vec_env_cls = SubprocVecEnv)
            env_eval = make_vec_env(EnergyPlusEnv, n_envs = n_envs, env_kwargs = env_kwargs, vec_env_cls = SubprocVecEnv)
            obs = env.reset()
            print("# observations = ", obs[0].shape)
            break

        except Exception as e:
            print("Error in creating environment: ", e)
    

    sim_days = 92

    n_days_eval = sim_days * args.num_episode
    n_steps_eval = sim_steps_per_day * n_days_eval

    
    learning_days = args.n_learning_days


    hyperparameters = {"ob_model_causality" : args.ob_model_causality, "start_month": args.start_month, "start_day": args.start_day,
                "end_month": args.end_month, "end_day": args.end_day,    
                "rl_option": args.rl_option, "occ_option": args.occ_option,
                "save_csv": args.save_csv, "save_trajectory": args.save_trajectory,
                "only_last_trajectory": args.only_last_trajectory, "model_name": args.model_name,
                "clean_scenarios": args.clean_scenarios, "print_values": args.print_values,
                "reset_previous_training": args.reset_previous_training,
                "log_tensorboard": args.log_tensorboard, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "gamma": args.gamma, "n_epochs": args.n_epochs,
                "num_episode": args.num_episode, "model_verbose": args.model_verbose,
                "n_nodes": args.n_nodes, "n_layers": args.n_layers, "learning_days": learning_days,
                "ent_coef" : ent_coef, "clip_range": clip_range, "vf_coef": vf_coef, "gae_lambda" :gae_lambda, "max_grad_norm": max_grad_norm,
                "r3_weight": args.obj_coeffs[0], "r4_weight": args.obj_coeffs[1],    
                "reward_form": args.reward_form, "occ_case": args.occ_case, 
                "ob_model": c15_ob_model
                }

    print(hyperparameters)

    hyperparameters_df = pd.DataFrame([hyperparameters])

    hyperparameters_df.to_csv("rl_results/{model_name}/hyperparameters.csv".format(model_name = args.model_name))

    n_steps = int(sim_steps_per_day * learning_days)

    eval_callback = EvalCallback(env_eval, best_model_save_path="./rl_results/{model_name}/".format(model_name = args.model_name),
                                log_path="./rl_results/{model_name}/".format(model_name = args.model_name), n_eval_episodes = 1 if args.occ_option == 0 else n_envs*2, 
                                eval_freq = n_steps_eval,
                                deterministic=True, render=False)
    

    if os.path.exists("rl_results/{model_name}/best_eval_reward.csv".format(model_name = args.model_name)):
        best_eval_reward_df = pd.read_csv("rl_results/{model_name}/best_eval_reward.csv".format(model_name = args.model_name))
        if not pd.isna(best_eval_reward_df["best_mean_reward"].max()):
            eval_callback.best_mean_reward = best_eval_reward_df["best_mean_reward"].max()
            print("Best mean reward found in previous training: ", eval_callback.best_mean_reward)

    print("- n_envs =", n_envs,    
        "\n- simulation days per episode =", sim_days, 
        "\n- simulation steps per episode | accross all envs  =", sim_days*sim_steps_per_day, sim_days*sim_steps_per_day*n_envs, "[steps]",
        "\n\n- steps of each learning/update = ", n_steps, "[steps]",
        "\n- days of each learning/update per env | accross all envs= ", learning_days, learning_days*n_envs, "[days]",
        "\n- steps of each evaluation per env | accross all env =", n_steps_eval, n_steps_eval * n_envs, "[steps]",
        "\n- days of each evaluation per env | accross all env=", n_days_eval, n_days_eval * n_envs, "[days]"
    )


    print("\n",progress," - Check training history (csv) file"); progress += 1

    print(args.start_month, args.start_day, args.end_month, args.end_day)

    policy_kwargs = dict(net_arch=[args.n_nodes for _ in range(args.n_layers)])
    
    class CustomCallback(BaseCallback):
        def __init__(self, eval_callback = None, verbose: int = 0):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.previous_best_reward = self.eval_callback.best_mean_reward
        def _on_training_start(self) -> None:
            pass

        def _on_rollout_start(self) -> None:
            current_step = self.num_timesteps

            if (current_step % (sim_days*sim_steps_per_day*n_envs) == 0) & (current_step != 0):
                
                if args.clean_scenarios == 1:
                    directory = "rl_results/{model_name}/scenarios/".format(model_name=args.model_name)
                    for root, dirs, files in os.walk(directory):
                        for dir in dirs:
                            shutil.rmtree(os.path.join(root, dir))

                latest_model_path = "rl_results/{}/latest_model.zip".format(args.model_name)
                self.model.save(latest_model_path)

                print(f"Plotting/Hyperparameter Update | Current experienced step: {current_step} | Current experienced days: {current_step/sim_steps_per_day}")

                colnames = ["reward", "timesteps", "time"]

                if os.path.exists("rl_results/{model_name}/monitor_logs/0.monitor.csv".format(model_name = args.model_name)):
                    
                    dfs = [None for _ in range(n_envs)]
                    fig, ax = plt.subplots(figsize=(10, 5))

                    try:
                        for i in range(n_envs):
                            dfs[i] = pd.read_csv("rl_results/{model_name}/monitor_logs/{i}.monitor.csv".format(model_name = args.model_name, i = i), names=colnames, on_bad_lines='skip')[2:]
                            dfs[i]["timesteps"] = dfs[i]["timesteps"].astype(int)
                            dfs[i]["time"] = dfs[i]["time"].astype(float)
                            dfs[i]["reward"] = dfs[i]["reward"].astype(float)                            
                            dfs[i] = dfs[i][dfs[i]["timesteps"] == int(sim_steps_per_day*sim_days)]
                            dfs[i]["accumulated_timesteps"] = dfs[i]["timesteps"].cumsum() 
                            dfs[i]["accumulated_days"] = dfs[i]["accumulated_timesteps"]/(sim_steps_per_day)
                            dfs[i]["Number of episodes"] = dfs[i]["accumulated_days"]/sim_days
                            dfs[i]["Training time [days]"] = dfs[i]["time"]/86400
                        
                            df = dfs[i]

                            df["Number of episodes_total"] = df["Number of episodes"]*n_envs

                            ax.scatter(df["Number of episodes_total"], df["reward"], color="C0", s = 10, alpha = 0.2)

                            n_windows = 10

                            if len(df["Number of episodes_total"]) > n_windows:
                                weights = np.ones(n_windows) / n_windows
                                sma = np.convolve(df["reward"].to_numpy(), weights, mode='valid')
                                ax.plot(df["Number of episodes_total"][n_windows-1:], sma, color="black", alpha = 0.2, label= f"Moving average ({n_windows*n_envs} episodes)")

                        if self.eval_callback is not None:
                            ax.text(0.5, 0.1, 
                                    f"Best Mean Reward: {self.eval_callback.best_mean_reward:.2f}",
                                    transform=ax.transAxes, fontsize=10)
                                                    
                            best_reward_log_path = "rl_results/{model_name}/best_eval_reward.csv".format(model_name=args.model_name)
                            
                            write_header = not os.path.exists(best_reward_log_path)

                            with open(best_reward_log_path, "a") as csv_file:
                                if write_header:
                                    csv_file.write("best_mean_reward,steps,days\n")
                                if self.eval_callback.best_mean_reward > self.previous_best_reward:
                                    csv_file.write(f"{self.eval_callback.best_mean_reward:.2f},{current_step},{current_step/sim_steps_per_day:.2f}\n")

                            self.previous_best_reward = self.eval_callback.best_mean_reward

                        ax.set_ylabel("Accumulated reward for training period [-]")
                        ax.set_xlabel("Number of episodes [-]")
                        secax = ax.secondary_xaxis('top', 
                                        functions = (lambda x: x * sim_days, lambda x: x / sim_days))
                        secax.set_xlabel('Observed days [-]')

                        fig.tight_layout()
                        fig.savefig("rl_results/{model_name}/reward_graph.png".format(model_name=args.model_name), dpi = 600)

                        plt.close()

                    except Exception as e:
                        print("Error in reward reporting/plotting: ", e)
                        plt.close()


        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if args.clean_scenarios == 1:
                directory = "rl_results/{model_name}/scenarios/".format(model_name=args.model_name)
                for root, dirs, files in os.walk(directory):
                    for dir in dirs:
                        shutil.rmtree(os.path.join(root, dir))
            pass

        def _on_training_end(self) -> None:
            pass


    if not os.path.exists("rl_results/{model_name}/latest_model.zip".format(model_name = args.model_name)):
        print("No previous model found. Training new model.")
        model = PPO(MlpPolicy, 
                    env = env, policy_kwargs=policy_kwargs, 
                    verbose=args.model_verbose, n_steps = n_steps, 
                    learning_rate = args.learning_rate, gamma = args.gamma,
                    n_epochs=args.n_epochs,
                    batch_size = args.batch_size,
                    ent_coef = ent_coef, clip_range = clip_range, vf_coef = vf_coef, gae_lambda = gae_lambda, max_grad_norm = max_grad_norm)
    else:
        print("Latest model found. Loading model.")
        model = PPO.load("rl_results/{model_name}/latest_model.zip".format(model_name = args.model_name),
                    env = env, policy_kwargs=policy_kwargs, 
                    verbose=args.model_verbose, n_steps = n_steps, 
                    learning_rate = args.learning_rate, gamma = args.gamma,
                    n_epochs=args.n_epochs,
                    batch_size = args.batch_size,
                    ent_coef = ent_coef, clip_range = clip_range, vf_coef = vf_coef, gae_lambda = gae_lambda, max_grad_norm = max_grad_norm)

    print("\n",progress," - Learn RL policy"); progress += 1
    
    model.learn(total_timesteps=n_steps_eval*100000000, log_interval = 1, callback = [eval_callback, CustomCallback(eval_callback= eval_callback)], progress_bar= True)

if __name__ == "__main__":
    while True:
        try:
            main()
        except EOFError as e:
            print("EOFError occurred: ", e)
