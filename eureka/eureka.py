import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
POKEMON_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../PokemonRedExperiments/baselines"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/pokemon'):
        env_parent = 'pokemon'
    elif f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac'):
        env_parent = 'isaac' 
    else:
        env_parent = 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    # set env_dir according to env_parent
    training_script = 'train.py'
    if env_parent == 'isaac':
        env_dir = ISAAC_ROOT_DIR
    elif env_parent == 'pokemon':
        env_dir = POKEMON_ROOT_DIR
        training_script = 'run_baseline_parallel_fast.py'
    else:
        env_dir = ISAAC_ROOT_DIR # default code just seemed to use ISAAC_ROOT_DIR
    output_file = f"{env_dir}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    if env_parent == 'pokemon':
        code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip_pokemon.txt')
        pokemon_codes = file_to_string(f'{prompt_dir}/pokemon_codes.txt')
        # only add this if gpt model context length is long enough:
        # TODO measure context length of gpt model
        # code_output_tip += pokemon_codes
    else:
        code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    if env_parent == 'pokemon':
        reward_signature = file_to_string(f'{prompt_dir}/reward_signature_general.txt')
    else:
        reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    create_task(env_dir, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample # if "gpt-3.5" in model else 4
        retry_delay = 10 #seconds
        retry_delay_exponent = 1.5

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(100):
                try:
                    logging.info(f"Iteration {iter}: Attempt {attempt+1} with chunk size {chunk_size}")
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= retry_delay_exponent
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")
        
        def pretty_print_obj(obj, file, indent=0, allow_value_linebreaks=True):
            for key, value in obj.items():
                if isinstance(value, dict):
                    file.write('  ' * indent + str(key) + ': \n')
                    pretty_print_obj(value, file, indent+1)
                else:
                    if allow_value_linebreaks and isinstance(value, str):
                        value = value.replace('\n', '\n' + '  ' * indent)
                    file.write('  ' * indent + str(key) + ': ' + str(value) + '\n')

            
        
        # write request + response to file for current iteration:
        with open(f"iter{iter}_response{total_samples-chunk_size}_to_{total_samples}.txt", 'w', encoding="utf-8") as file:
            # pretty print request messages to text file
            file.write("\n========================================================Messages:\n\n\n")
            for message in messages:
                file.write("\n--------------------------------------------------------\n\n")
                pretty_print_obj(message, file)
            
            file.write("\n========================================================Responses:\n\n\n")
            for response in responses:
                file.write("\n--------------------------------------------------------\n\n")
                pretty_print_obj(response, file)

        


        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    
            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature! {code_string}")
                continue

            code_runs.append(code_string)
            if (env_parent == 'pokemon'):
                reward_signature = [
                    f"total_reward, sub_rewards = {gpt_reward_signature}",
                ]
            else:
                reward_signature = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot find compute_reward function in the environment code! {task_code_string}")
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w', encoding="utf-8") as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                if "@torch.jit.script" not in code_string and env_parent != 'pokemon':
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w', encoding="utf-8") as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            logging.info(f"Iteration {iter}: Code Run {response_id} saved to {output_file}")
            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w', encoding="utf-8") as f:
                if env_parent == 'pokemon':
                    process = subprocess.Popen([
                            'python', '-u', f'{env_dir}/{training_script}',
                            'hydra/output=subprocess',
                            f'task={task}{suffix}',
                            #f'headless={cfg.headless}',
                            #f'save_video={cfg.capture_video}',
                            f'gym_env=env_iter{iter}_response{response_id}',
                        ],
                        stdout=f, stderr=f)
                else:
                    process = subprocess.Popen([
                            'python', '-u', f'{env_dir}/{training_script}',  
                            'hydra/output=subprocess',
                            f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                            f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                            f'max_iterations={cfg.max_iterations}'
                        ],
                        stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
        
        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            print(f"Code Run {response_id} executing...")
            rl_run.communicate()
            print(f"Code Run {response_id} finished!")
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r', encoding="utf-8") as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            tensorboard_logs = {}
            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                if env_parent == 'pokemon':
                    tensorboard_logs = {}
                    for i, line in enumerate(lines):
                        if line.startswith('step:'):
                            # split on whitespace to get key-value pairs
                            # keys on even indices, values on odd indices
                            # convert to dict
                            # for each key, append to the tensor in tensorboard_logs
                            items = [item.strip().strip(':').strip('-') for item in line.split()]
                            keys = items[::2]
                            # wrap each value in a list
                            values = [x for x in items[1::2]]
                            # aggressively extract the float number from string sentence (one per x).  0 if fail
                            values = [float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0]) if len(re.findall(r"[-+]?\d*\.\d+|\d+", x)) > 0 else 0 for x in values]
                            # append to tensorboard_logs
                            for key, value in zip(keys, values):
                                if key not in tensorboard_logs:
                                    tensorboard_logs[key] = []
                                tensorboard_logs[key].append(value)
                    # delete tensorboard_logs['step']
                    if ('step' in tensorboard_logs):
                        del tensorboard_logs['step']
                    tensorboard_logs['gt_reward'] = tensorboard_logs['sum']
                    tensorboard_logs['gpt_reward'] = tensorboard_logs['sum']
                    tensorboard_logs['consecutive_successes'] = [max(tensorboard_logs['sum']) // min(tensorboard_logs['sum'])]
                else:
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)


                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)
            logging.info(f"Iteration {iter}: Code Run {response_id} finished! Success rate: {successes[-1]}, Sum: {tensorboard_logs.get('sum', [0])[-1]}")
            # logging.info(content)

        
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info(contents)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        logging.info(f"Iteration {iter}: Successes out of {cfg.sample} samples: {np.sum(np.array(successes) >= 0.)}")
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
            
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w', encoding="utf-8") as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w', encoding="utf-8") as f:
            if env_parent == 'pokemon':
                process = subprocess.Popen([
                        'python', '-u', f'{env_dir}/{training_script}',
                        'hydra/output=subprocess',
                        f'task={task}{suffix}',
                        #f'headless={cfg.headless}',
                        #f'save_video={cfg.capture_video}',
                        f'gym_env=env_iter{iter}_response{response_id}',
                    ],
                    stdout=f, stderr=f)
            else:
                process = subprocess.Popen(['python', '-u', f'{env_dir}/train.py',  
                        'hydra/output=subprocess',
                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                    ],
                    stdout=f, stderr=f)

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        print(f"Reward code Run {i} executing...")
        rl_run.communicate()
        print(f"Reward code Run {i} finished!")
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r', encoding="utf-8") as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        if env_parent == 'pokemon':
            tensorboard_logs = {}
            for i, line in enumerate(lines):
                if line.startswith('step:'):
                    # split on whitespace to get key-value pairs
                    # keys on even indices, values on odd indices
                    # convert to dict
                    # for each key, append to the tensor in tensorboard_logs
                    items = [item.strip().strip(':').strip('-') for item in line.split()]
                    keys = items[::2]
                    # wrap each value in a list
                    values = [float(x) for x in items[1::2]]
                    # append to tensorboard_logs
                    for key, value in zip(keys, values):
                        if key not in tensorboard_logs:
                            tensorboard_logs[key] = []
                        tensorboard_logs[key].append(value)
            tensorboard_logs['gt_reward'] = tensorboard_logs['sum']
            tensorboard_logs['gpt_reward'] = tensorboard_logs['sum']
            #tensorboard_logs['consecutive_successes'] = 
        else:
            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()