import os
import platform
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
# Gym/numpy 내부 deprecation (e.g. rng.randn) 억제
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*randn.*')
# 서드파티 deprecation 억제 (Cython, wandb, gym, importlib 등)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='Cython')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='distutils')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='wandb')
warnings.filterwarnings('ignore', message='.*load_module.*')
warnings.filterwarnings('ignore', message='.*dep_util.*')
warnings.filterwarnings('ignore', message='.*start_method.*')
warnings.filterwarnings('ignore', message='.*app_url.*')
warnings.filterwarnings('ignore', message=r'.*Scope\.user.*')
warnings.filterwarnings('ignore', message='.*[Gg]ym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*upgrade to Gymnasium.*')
warnings.filterwarnings('ignore', category=UserWarning, module='wandb')

import json
import random
import time
from datetime import datetime

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.mem_utils import get_memory_metrics, get_memory_summary

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_boolean('time_logging', False, 'Whether to log per-step training/inference time.')
flags.DEFINE_boolean('mem_logging', False, 'Whether to log peak GPU (JAX allocator) and host RAM usage.')
flags.DEFINE_boolean('use_wandb', True, 'Whether to log to wandb.')

config_flags.DEFINE_config_file('agent', 'agents/aligen.py', lock_config=False)


def main(_):
    # Load agent config first (used for run name).
    config = FLAGS.agent

    # Set up logger.
    # Run name format: algorithm_name_env_name_seed_YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hp_tag = f'_a{config["alpha"]}' if 'alpha' in config else ''
    hp_tag += f'_t{config["temp"]}' if 'temp' in config else ''
    hp_tag += f'_it{config["inv_temp"]}' if 'inv_temp' in config else ''
    exp_name = f'{config["agent_name"]}_{FLAGS.env_name}{hp_tag}_{FLAGS.seed}_{timestamp}'
    if FLAGS.use_wandb:
        setup_wandb(project='aligen', group=FLAGS.run_group, name=exp_name)
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    else:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            time_prefix = 'time/offline'
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            time_prefix = 'time/online'
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/train_per_step(ms)'] = ((time.time() - last_time) / FLAGS.log_interval) * 1000
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            if FLAGS.mem_logging:
                train_metrics.update(get_memory_metrics())
            last_time = time.time()
            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            if FLAGS.time_logging:
                eval_start = time.time()
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            if FLAGS.time_logging:
                eval_time = time.time() - eval_start
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v
            if FLAGS.time_logging:
                total_eval_steps = sum(len(t['reward']) for t in trajs)
                eval_time_ms = eval_time * 1000
                eval_metrics[f'{time_prefix}_eval_total(ms)'] = eval_time_ms
                eval_metrics[f'{time_prefix}_eval_per_episode(ms)'] = eval_time_ms / FLAGS.eval_episodes
                eval_metrics[f'{time_prefix}_infer_per_step(ms)'] = eval_time_ms / total_eval_steps if total_eval_steps > 0 else 0

            if FLAGS.use_wandb and FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.use_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            
            # 훈련 시간에 평가 시간이 포함되지 않도록 기준 시간 초기화
            last_time = time.time()

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()

    if FLAGS.mem_logging:
        mem_summary = get_memory_summary()
        mem_summary['agent_name'] = config['agent_name']
        mem_summary['env_name'] = FLAGS.env_name
        mem_summary['seed'] = FLAGS.seed
        mem_summary['offline_steps'] = FLAGS.offline_steps
        with open(os.path.join(FLAGS.save_dir, 'mem_summary.json'), 'w') as f:
            json.dump(mem_summary, f, indent=2)
        gpu_peak = mem_summary.get('gpu_peak_MiB')
        gpu_peak_str = f'{gpu_peak:.1f} MiB' if gpu_peak is not None else 'n/a'
        print(
            f'[mem] {config["agent_name"]} {FLAGS.env_name} seed={FLAGS.seed} '
            f'gpu_peak={gpu_peak_str} host_ram_peak={mem_summary["host_ram_peak_GB"]:.2f} GB'
        )


if __name__ == '__main__':
    app.run(main)
