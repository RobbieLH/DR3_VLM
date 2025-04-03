from PIL import Image  #robbie added at 2025-3-19
# import VLM.Qwen #robbie add
# from VLM.sbert import SentenceBert  #robbie add
import gym
import numpy as np

global_counter = 0

class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

         ##robbie2025-3-17 add VLM
        # self.vlm = VLM.Qwen.QwenVL("3B")
        # self.sbert = SentenceBert()
         ##robbie2025-3-17 add VLM
        self.envVLM_count = 0

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        ##robbie add
        spaces["text_embed"] = gym.spaces.Box(-np.inf, np.inf, (384,), dtype=np.float32)
        ##robbie add
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["text_embed"] = self.VLMtext(obs["image"] )
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["text_embed"] = self.VLMtext(obs["image"] )
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
    
    def VLMtext(self, img):          
        img_path = "tmp_img.png"
        global global_counter
        # 计数器加 1
        global_counter = global_counter + 1
      
        if global_counter % 8 == 9:
           img = Image.fromarray(img)
       
        # img.save(img_path)
        #    text = self.vlm(img)
        #    text_embed = self.sbert(text)
        #    text_embed = np.array(text_embed)
        else:
           text_embed = np.zeros(384)
        # print(f"vlm调用的次数: {global_counter}")
       
        # text = '这张图片显示了一个卡通风格的简笔画人物。这个人物有一个长脖子和一个圆圆的身体，没有明显的四肢或手部特征。背景是浅色的方格图案，可能代表地板或地面。整体风格简洁且具有一定的抽象感。'
        

        self.envVLM_count += 1
        if self.envVLM_count == 1:
            img_np = np.array(img)
            print("图片形状（NumPy）：", img_np.shape)
            print("图片形状（NumPy）：", text_embed.shape)
            print("\n我到这里了，hahah\n")
        return text_embed
