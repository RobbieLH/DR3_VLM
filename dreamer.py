from PIL import Image  #robbie added at 2025-3-17
import argparse
import functools
import os
import pathlib
import sys
import gym
import pdb

# 设置 MUJOCO_GL 环境变量为 osmesa。OSMesa图形库的一个变体允许在没有显示器下进行图形渲染。
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

# 将当前文件所在目录添加到系统路径中，以便后续导入自定义模块
sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

##robbie added
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)



to_np = lambda x: x.detach().cpu().numpy()

#  用于实现一个强化学习智能体。该智能体类
#  整合了世界模型、任务行为模型和探索行为模型
#  等组件，具备训练、生成动作策略以及记录训练指标等功能
class Dreamer(nn.Module):
    # 主要用于初始化智能体的各种属性和组件，
    # 包括配置信息、日志记录器、训练和探索相关的
    # 判断条件、世界模型、行为模型等
    def __init__(self, obs_space, act_space, config, logger, dataset):
         # 调用父类的构造函数
        super(Dreamer, self).__init__()
        # 保存配置信息
        self._config = config
        # 保存日志记录器
        self._logger = logger
        # 用于判断是否应该记录日志，每 config.log_every 步记录一次
        self._should_log = tools.Every(config.log_every)
        # 计算每批数据的步数
        batch_steps = config.batch_size * config.batch_length
         # 用于判断是否应该进行训练，根据训练比率决定
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        # 用于判断是否应该进行预训练，仅执行一次
        self._should_pretrain = tools.Once()
         # 用于判断是否应该重置环境，每 config.reset_every 步重置一次
        self._should_reset = tools.Every(config.reset_every)
        # 用于判断是否处于探索阶段，直到 config.expl_until 步结束
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        # 用于存储训练指标
        self._metrics = {}
        # this is update step # 计算当前的更新步骤
        self._step = logger.step // config.action_repeat
        # 记录更新次数
        self._update_count = 0
        # 保存数据集
        self._dataset = dataset
         # 初始化世界模型
        # #robbie added
        # if self._config.use_vlm and "text_embed" not in obs.keys():
        


        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
         # 初始化任务行为模型
        self._task_behavior = models.ImagBehavior(config, self._wm)
         # 如果配置中启用了编译且不是 Windows 系统，则对模型进行编译
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
         # 定义奖励函数
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        # 根据配置选择探索行为模型
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

            
        #robbie
        self.call_count = 0

    # 主要实现了智能体在训练模式和非训练模式下的行为，
    # 包括训练智能体、记录训练指标、生成动作策略以及更新状态等功能
    def __call__(self, obs, reset, state=None, training=True):
        # 获取当前的更新步骤
        step = self._step
        
        #robbie add
        self.call_count += 1
        if self.call_count == 1:
            #print(f"第{self.call_count}次调用的，__call__ 方法中 obs 的数据为: {obs}")
            logging.info(f"第{self.call_count}次调用的，__call__ 方法中 obs 的数据类型为: {type(obs)}")
            logging.info(f"第{self.call_count}次调用的，__call__ 方法中 obs 的数据为: {obs.keys()}")
            logging.info(f"第{self.call_count}次调用的，__call__ 方法中 obs中的image数组形状的数据为: {obs['image'].shape}")
            print(f"第{self.call_count}次调用的，__call__ 方法中 obs 的数据类型为: {type(obs)}")
            print(f"第{self.call_count}次调用的，__call__ 方法中 obs 的数据为: {obs.keys()}")
            print(f"第{self.call_count}次调用的，__call__ 方法中 obs中的image数组形状的数据为: {obs['image'].shape}")

        # 假设obs是包含图像数据的字典
            img_data = obs['image'][0]
            # 将图像数据转换为PIL图像对象
            img = Image.fromarray(np.uint8(img_data))
            # 保存图像到本地
           

            current_dir = os.getcwd()
            # 构建保存文件的完整路径
            file_name = f'output_image_{self.call_count}.jpg'
            save_path = os.path.join(current_dir, file_name)

            # 保存图像到当前目录
            img.save(save_path)
            logging.info(f"图像已保存到: {save_path}")
            print(f"图像已保存到: {save_path}")

             #robbie add
             
            

            print(f"\n{self._config.use_vlm}\n")
        # ##robbie2025-3-17 add VLM
        # if self._config.use_vlm and "text_embed" not in obs.keys():
            
        #     text = []
        #     for i in range(len(obs['image'])):
        #         img = obs['image'][i]
        #         img_path = "tmp_img.png"
        #         img = Image.fromarray(img)
        #         # img.save(img_path)
        #         t = self._wm.vlm(img)
        #         text.append(t)
        #         if self.call_count == 1:
        #             print("\n到1面了\n")
        #     text_embed = self._wm.sbert(text)
        #     if self.call_count == 1:
        #             print("\n到2面了\n")
        #     text_embed = np.array(text_embed)

        #     if self.call_count == 1:
        #         min_value = np.min(text_embed)
        #         max_value = np.max(text_embed)
        #         shape = text_embed.shape
        #         dtype = text_embed.dtype
        #         print(f"text_embed 的最小值: {min_value}")
        #         print(f"text_embed 的最大值: {max_value}")
        #         print(f"text_embed 的形状: {shape}")
        #         print(f"text_embed 的数据类型: {dtype}")

        #     obs['text_embed'] = text_embed
        #     print(f"第{self.call_count}次调用的，添加完之后 obs 的数据为: {obs.keys()}")
        #     if self.call_count == 1:
        #             print("\n到3面了\n")
          ###robbie2025-3-17 add VLM
        # import pdb  
        # pdb.set_trace() 

        # 如果处于训练模式
        if training:
            # 计算需要进行训练的步数
            steps = (
                # 如果需要进行预训练，则执行预训练步数
                self._config.pretrain
                if self._should_pretrain()
                # 否则，根据训练判断条件决定训练步数
                else self._should_train(step)
            )
            # 循环执行训练步骤
            for _ in range(steps):
                # 从数据集中获取下一个批次的数据并进行训练
                self._train(next(self._dataset))
                 # 训练更新次数加 1
                self._update_count += 1
                # 记录当前的训练更新次数到指标字典中
                self._metrics["update_count"] = self._update_count
            # 如果满足日志记录条件
            if self._should_log(step):
                 # 遍历所有的训练指标
                for name, values in self._metrics.items():
                    # 计算指标的平均值并记录到日志中
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                     # 如果配置中启用了视频预测日志记录
                if self._config.video_pred_log:
                    # 进行视频预测
                    openl = self._wm.video_pred(next(self._dataset))
                    # 将视频预测结果记录到日志中
                    self._logger.video("train_openl", to_np(openl))
                    # 写入日志，同时记录每秒帧数（fps）
                self._logger.write(fps=True)
         # 调用 _policy 方法生成策略输出和状态
        policy_output, state = self._policy(obs, state, training)
        # 如果处于训练模式
        if training:
             # 更新当前的步骤数，根据 reset 的长度增加
            self._step += len(reset)
             # 更新日志记录器的步数，考虑动作重复次数
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

        # _policy 函数通过一系列步骤，根据观测和状态生成动作，
        # 并更新智能体的内部状态。在不同的模式和阶段，采用不同的
        # 动作生成策略，以适应训练和评估的需求。同时，对动作和状态
        # 进行了必要的处理和梯度分离，确保训练过程的稳定性和有效性。
    def _policy(self, obs, state, training):
        if state is None:
             # 如果状态为空，则初始化潜在状态和动作
            latent = action = None
        else:
            # 从状态中解包潜在状态和动作
            latent, action = state
            # 对观测进行预处理
        obs = self._wm.preprocess(obs)
         # 通过编码器获取嵌入表示
        embed = self._wm.encoder(obs)
        # 通过动态模型更新潜在状态
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        # 如果配置中启用了评估状态均值，则使用均值作为潜在状态
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
            # 获取特征表示，一般进行决策和奖励
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            # 在非训练模式下，使用任务行为模型的动作模式
            actor = self._task_behavior.actor(feat)
            # 返回该分布的众数
            action = actor.mode()
        elif self._should_expl(self._step):
            # 在探索阶段，使用探索行为模型的动作采样
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            # 在非探索阶段，使用任务行为模型的动作采样
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        # 计算动作的对数概率
        logprob = actor.log_prob(action)
        # 分离潜在状态和动作的梯度
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        # 如果动作分布为 onehot_gumble，则将动作转换为 one-hot 编码
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        # 构建策略输出    
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state


    # 主要功能是进行强化学习智能体的训练，
    # 包括训练世界模型、任务行为模型以及
    # 探索行为模型，并记录相关的训练指标
    def _train(self, data):
        # 用于存储训练指标
        metrics = {}
        # 训练世界模型，获取后验状态、上下文和指标
        post, context, mets = self._wm._train(data)
        # 更新指标
        metrics.update(mets)
        # 设置起始状态
        start = post
         # 定义奖励函数???这里的f,s,a 分别是啥？
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
          # 训练任务行为模型，更新指标
        metrics.update(self._task_behavior._train(start, reward)[-1])
        # 如果探索行为不是贪婪策略，则训练探索行为模型
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
             # 将指标添加到全局指标字典中
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

     ###
     #主要功能是统计指定文件夹中
     # 所有符合条件的 .npz 文件所代表的步数总和
     ###
def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

        ###
        #主要功能是根据传入的配置参数 config、
        # 环境模式 mode 和环境实例编号 id，
        # 创建不同类型的强化学习环境，
        # 并对创建的环境应用一系列的包装器（wrapper）
        # 来进行预处理和功能扩展
        ###

def make_env(config, mode, id):
    # 从配置信息中提取任务套件和具体任务名称
     # 假设 config.task 格式为 "suite_task"，通过 split("_", 1) 以 "_" 为分隔符，最多分割一次
    suite, task = config.task.split("_", 1)

    # 如果任务套件是 "dmc"（DeepMind Control）
    if suite == "dmc":
         # 导入 dmc 环境模块
        import envs.dmc as dmc

        # 创建 DeepMind Control 环境实例
        # task 是具体任务名称，config.action_repeat 是动作重复次数，config.size 是环境图像尺寸
        # seed 是随机数种子，使用 config.seed 加上 id 确保每个环境实例的随机性不同
        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
         # 使用 NormalizeActions 包装器对环境动作进行归一化处理
        env = wrappers.NormalizeActions(env)
     # 如果任务套件是 "atari"（Atari 游戏）
    elif suite == "atari":
         # 导入 atari 环境模块
        import envs.atari as atari

        # 创建 Atari 环境实例
        # 传入各种配置参数，如动作重复次数、图像尺寸、是否使用灰度图像等
        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,     #使用 OpenCV 库来完成图像尺寸
            seed=config.seed + id,
        )
        # 使用 OneHotAction 包装器将动作转换为独热编码
        env = wrappers.OneHotAction(env)
        # 如果任务套件是 "dmlab"
    elif suite == "dmlab":
        # 导入 dmlab 环境模块
        import envs.dmlab as dmlab
        
        # 创建 DeepMind Labyrinth 环境实例
        # 根据 mode 是否包含 "train" 确定使用训练模式还是测试模式
        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
        # 如果任务套件是 "memorymaze"
    elif suite == "memorymaze":
        # 从 envs.memorymaze 模块导入 MemoryMaze 类
        from envs.memorymaze import MemoryMaze

        # 创建 MemoryMaze 环境实例
        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
         # 导入 crafter 环境模块
        import envs.crafter as crafter

        # 创建 Crafter 环境实例
        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft
        # 创建 Minecraft 环境实例
        # 传入任务名称、图像尺寸和破坏速度等配置参数
        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    
    # 使用 TimeLimit 包装器为环境设置时间限制，超过时间限制则终止环境交互
    env = wrappers.TimeLimit(env, config.time_limit)
    # 使用 SelectAction 包装器选择指定键的动作，这里选择 "action" 键对应的动作
    env = wrappers.SelectAction(env, key="action")
    # 使用 UUID 包装器为环境添加唯一标识符
    env = wrappers.UUID(env)
    if suite == "minecraft":
        # 使用 RewardObs 包装器将奖励信息添加到观测中
        env = wrappers.RewardObs(env)
    return env

              ###
              # 主要实现了设置实验环境、
              # 加载和预处理数据、创建智能体和环境、
              # 预填充数据集、进行训练和评估等功能
              ###
def main(config):
    # 设置全局随机种子，确保实验的可重复性
    tools.set_seed_everywhere(config.seed)
     # 如果配置要求确定性运行，则启用相关设置
    if config.deterministic_run:
        tools.enable_deterministic_run()
    # 将配置中的日志目录路径进行扩展，转换为绝对路径
    logdir = pathlib.Path(config.logdir).expanduser()
    # 如果训练数据目录未指定，则使用日志目录下的 train_eps 目录
    config.traindir = config.traindir or logdir / "train_eps"
    # 如果评估数据目录未指定，则使用日志目录下的 eval_eps 目录
    config.evaldir = config.evaldir or logdir / "eval_eps"
     # 由于动作重复机制，对总步数进行调整
    config.steps //= config.action_repeat
    # 调整评估频率，考虑动作重复
    config.eval_every //= config.action_repeat
    # 调整日志记录频率，考虑动作重复
    config.log_every //= config.action_repeat
     # 调整环境的时间限制，考虑动作重复
    config.time_limit //= config.action_repeat

    # 打印日志目录路径
    print("Logdir", logdir)
    # logging.info(f"Logdir:{logdir}")
     # 创建日志目录，如果父目录不存在则一并创建，若目录已存在则不报错
    logdir.mkdir(parents=True, exist_ok=True)
    # 创建训练数据目录，如果父目录不存在则一并创建，若目录已存在则不报错
    config.traindir.mkdir(parents=True, exist_ok=True)
     # 创建评估数据目录，如果父目录不存在则一并创建，若目录已存在则不报错
    config.evaldir.mkdir(parents=True, exist_ok=True)
    # 统计训练数据目录中的步数
    step = count_steps(config.traindir)
    # step in logger is environmental step
    # 初始化日志记录器，传入日志目录和当前步数（考虑动作重复）
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)
     # 打印提示信息，表示开始创建环境
    print("Create envs.")
    # logging.info(f"Create envs.")
    # 如果指定了离线训练数据目录，则使用该目录
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    # 从指定目录加载训练剧集数据，限制加载的数量
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    # 打印动作空间信息
    print("Action Space", acts)
    # logging.info(f"Action Space:{acts}")
     # 确定动作数量，如果动作空间是离散的，则取动作数量；否则取动作向量的维度
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    # 如果没有指定离线训练数据目录
    if not config.offline_traindir:
         # 计算需要预填充数据集的步数
        prefill = max(0, config.prefill - count_steps(config.traindir))
        # 打印预填充数据集的提示信息
        print(f"Prefill dataset ({prefill} steps).")
        # 创建一个服从均匀分布的随机动作生成器，为强化学习智能体在探
        # 索环境初期提供随机动作，帮助智能体快速收集环境信息，
        # 填充数据集，为后续的训练做好准备。
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )
        # 定义随机智能体：
        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        # 模拟随机智能体与环境交互，填充数据集
        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")
    logging.info("Simulate agent.")
    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    
    ##robbie
    # 定义 text_embed 的空间
    # text_embed_space = gym.spaces.Box(-float('inf'), float('inf'), (384,), dtype='float32')

    # # 创建新的 observation_space
    # obs_space = gym.spaces.Dict({
    #     **train_envs[0].observation_space,
    #     'text_embed': text_embed_space
    #     })

    # print("在agent上面：", obs_space)
    # ##robbie


    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    
    # #打印观测空间observation_space
    # print("\ntrain_envs[0].observation_space是", type(obs_space))
    #     # 遍历 Dict 空间的所有键
    # for key in obs_space.spaces.keys():
    #     print(f"键 {key} 的数据类型是: {type(key)}")
    # # 获取 height 键对应的值
    # height_space = obs_space['height']
    # print(f"height 对应的值的数据类型是: {type(height_space)}")
    # print("\ntrain_envs[0].observation_space是", type(obs_space['height']))
    print("\ntrain_envs[0].observation_space是", train_envs[0].observation_space)
    # logging.info(f"train_envs[0].observation_space是, {train_envs[0].observation_space}")
     # 暂时禁用智能体参数的梯度计算
    agent.requires_grad_(requires_grad=False)
    # 如果存在最新的检查点文件
    if (logdir / "latest.pt").exists():
         # 加载检查点文件
        checkpoint = torch.load(logdir / "latest.pt")
        # 加载智能体的状态字典
        agent.load_state_dict(checkpoint["agent_state_dict"])
        # 递归加载优化器的状态字典
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # 标记智能体不需要再进行预训练
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    # 确保在总步数加上评估间隔步数内，评估操作至少执行一次,agent._step只在训练阶段改变
    while agent._step < config.steps + config.eval_every:
        # 写入日志
        logger.write()
        # 如果配置的评估剧集数量大于 0
        if config.eval_episode_num > 0:
            # 打印开始评估的提示信息
            print("Start evaluation.")
            # 创建一个部分应用的评估策略函数，禁用训练模式
            eval_policy = functools.partial(agent, training=False)
             # 使用评估策略模拟环境交互进行评估
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            # 如果配置要求记录视频预测结果
            if config.video_pred_log:
                # 生成视频预测结果
                video_pred = agent._wm.video_pred(next(eval_dataset))
                # 将视频预测结果记录到日志中
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
         # 使用智能体模拟环境交互进行训练
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
         # 准备要保存的项目，包括智能体状态字典和优化器状态字典
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
         # 保存检查点文件
        torch.save(items_to_save, logdir / "latest.pt")
        # 关闭所有训练和评估环境
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

###这段代码的主要功能是解析命令行参数，
# 读取配置文件，并根据配置文件和命令行
# 参数更新配置信息，最后调用 main 函数启动程序
# ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    
    main(parser.parse_args(remaining))
