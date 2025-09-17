# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""  # 使用RSL-RL训练强化学习智能体的脚本

"""Launch Isaac Sim Simulator first."""  # 请先启动Isaac Sim仿真器


# 标准库导入
import argparse  # 用于命令行参数解析
import sys       # 用于操作Python运行时环境


# Isaac Lab仿真应用启动器 自带的
from isaaclab.app import AppLauncher


# 本地模块导入
import cli_args  # isort: skip  # 用于添加和处理自定义命令行参数


# ========== 命令行参数设置 ==========
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")  # 是否在训练过程中录制视频
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")  # 单个视频的步数
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")  # 两次视频录制之间的步数间隔
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")  # 并行仿真环境数量
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 任务名称
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")  # 随机种子
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")  # 最大训练迭代次数
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")  # wandb注册表名（用于下载动作数据）

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)  # 添加RSL-RL相关参数
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)  # 添加仿真应用相关参数
args_cli, hydra_args = parser.parse_known_args()  # 解析参数，args_cli为主参数，hydra_args为Hydra配置参数

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True  # 如果需要录制视频，则强制启用仿真环境摄像头

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args  # 清理sys.argv，避免Hydra参数冲突

# launch omniverse app
app_launcher = AppLauncher(args_cli)  # 启动仿真应用（如Omniverse/Isaac Sim）
simulation_app = app_launcher.app


# ========== 其余依赖导入 ==========


import gymnasium as gym  # RL环境接口
import os                # 文件与路径操作
import torch             # 深度学习库
from datetime import datetime  # 时间戳生成


# Isaac Lab环境与工具
from isaaclab.envs import (
    DirectMARLEnv,           # 多智能体环境
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,    # 管理型RL环境配置
    multi_agent_to_single_agent,  # 多智能体转单智能体
)
from isaaclab.utils.dict import print_dict         # 字典美观打印
from isaaclab.utils.io import dump_pickle, dump_yaml  # 配置保存
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # PPO配置与环境包装
from isaaclab_tasks.utils import get_checkpoint_path  # 检查点路径获取
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra任务装饰器


# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401  # 注册自定义任务
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # 自定义PPO Runner


# CUDA相关设置，提高训练速度
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False



# ========== 训练主流程 ==========
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""  # 使用RSL-RL算法训练智能体
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)  # 用命令行参数覆盖Hydra配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed  # 设置环境随机种子
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device  # 设置设备

    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"  # 若未指定版本，默认用:latest
    import pathlib

    import wandb
    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")  # 下载动作数据文件，写入环境配置

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)  # 创建gym环境，传入配置
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 每隔video_interval步录一次
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境用于视频录制

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # 如果是多智能体环境，转为单智能体

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # 用RslRlVecEnvWrapper包装环境，适配RSL-RL算法的训练流程和接口

    # create runner from rsl-rl 实际上用的是MotionOnPolicyRunner
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )  # 创建PPO Runner
    # write git state to logs
    runner.add_git_repo_to_log(__file__)  # 记录当前git仓库状态到日志
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)  # 恢复训练

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)  # 保存环境配置
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)  # 保存智能体配置
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)  # 开始训练

    # close the simulator
    env.close()  # 关闭环境



# ========== 程序入口 ==========
if __name__ == "__main__":
    # run the main function
    main()  # 运行主函数，开始训练
    # close sim app
    simulation_app.close()  # 关闭仿真应用
