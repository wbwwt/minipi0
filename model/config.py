# import dataclasses
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Tuple, Any

# from transformers import PretrainedConfig

# # --- 常量定义 ---
# DEFAULT_IMAGE_SIZE = 224

# @dataclass
# class RTCConfig:
#     """实时控制配置 (保留此类以免加载旧 Config 报错，虽暂不启用)"""
#     enabled: bool = False
#     name: str = "frequency_pd"
#     P: float = 0.05
#     D: float = 0.005
#     target_frequency: float = 15.0

# @dataclass
# class PI0Config(PretrainedConfig):
#     model_type = "pi0"
    
#     # --- 1. 基础模型配置 ---
#     paligemma_variant: str = "gemma_300m"
#     action_expert_variant: str = "gemma_300m"
    
#     # 图像分辨率 (Height, Width)
#     image_resolution: Tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    
#     # --- 2. 核心维度配置 (必须与训练数据一致) ---
#     # 状态维度 (你的机械臂关节数 + 夹爪)
#     max_state_dim: int = 7  # 例如: 6关节 + 1夹爪
#     # 动作维度
#     max_action_dim: int = 7 
    
#     # --- 3. 序列与时间相关 ---
#     # 每次预测多少步 (Chunk Size)
#     chunk_size: int = 50
#     # 推理时实际执行多少步
#     n_action_steps: int = 10  # 通常小于 chunk_size
    
#     # 扩散/流匹配参数
#     num_inference_steps: int = 10
#     min_period: float = 0.01
#     max_period: float = 1000.0
    
#     # 时间采样参数 (Training only)
#     time_sampling_beta_alpha: float = 1.0
#     time_sampling_beta_beta: float = 1.0
#     time_sampling_scale: float = 1.0
#     time_sampling_offset: float = 0.0
    
#     # --- 4. 训练策略参数 ---
#     freeze_vision_encoder: bool = False
#     train_expert_only: bool = False
#     gradient_checkpointing: bool = False
#     dtype: str = "bfloat16" # "float32" or "bfloat16"
    
#     # --- 5. 特征描述 (用于自动推断维度，这里留空或手动指定) ---
#     # 这是一个简化版的 input_features，不再依赖 LeRobot 的复杂 schema
#     input_features: Dict[str, Any] = field(default_factory=lambda: {
#         "observation.images.cam_high": {"shape": (3, 224, 224), "dtype": "float32"},
#         "observation.state": {"shape": (7,), "dtype": "float32"},
#         "observation.language_instruction": {"shape": (1,), "dtype": "string"}
#     })
    
#     output_features: Dict[str, Any] = field(default_factory=lambda: {
#         "action": {"shape": (7,), "dtype": "float32"}
#     })
    
#     # --- 6. 归一化映射 (极其重要，用于推理时反归一化) ---
#     # 这里存储的是 key 到 normalization mode 的映射
#     normalization_mapping: Dict[str, str] = field(default_factory=lambda: {
#         "observation.images.cam_high": "identity", # 图像通常由 transform 处理
#         "observation.state": "mean_std",
#         "action": "mean_std",
#     })
    
#     # --- 7. RTC 配置 ---
#     rtc_config: Optional[RTCConfig] = None
    
#     # --- 8. 编译选项 ---
#     compile_model: bool = False
#     compile_mode: str = "reduce-overhead"
    
#     # 必须的初始化函数，用于接收 **kwargs
#     def __init__(self, **kwargs):
#         # 提取 input_features 和 output_features 中可能存在的维度信息
#         # 以覆盖 max_state_dim 和 max_action_dim
#         if "input_features" in kwargs:
#             feats = kwargs["input_features"]
#             if "observation.state" in feats:
#                 self.max_state_dim = feats["observation.state"]["shape"][0]
        
#         if "output_features" in kwargs:
#             feats = kwargs["output_features"]
#             if "action" in feats:
#                 self.max_action_dim = feats["action"]["shape"][0]
                
#         # 处理 rtc_config 从 dict 转为对象 (如果从 json 加载)
#         if "rtc_config" in kwargs and isinstance(kwargs["rtc_config"], dict):
#             kwargs["rtc_config"] = RTCConfig(**kwargs["rtc_config"])

#         super().__init__(**kwargs)

#     # 兼容性函数：模仿 LeRobot 的 validate_features，这里什么都不做或仅做简单检查
#     def validate_features(self):
#         pass


from typing import Dict, List, Optional, Tuple, Any
from transformers import PretrainedConfig

# --- 常量定义 ---
DEFAULT_IMAGE_SIZE = 224

class PI0Config(PretrainedConfig):
    model_type = "pi0"
    
    def __init__(
        self,
        # 1. 基础模型配置
        paligemma_variant: str = "gemma_300m",
        action_expert_variant: str = "gemma_300m",
        image_resolution: Tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        
        # 2. 核心维度配置
        max_state_dim: int = 7,
        max_action_dim: int = 7,
        
        # 3. 序列与时间相关
        chunk_size: int = 50,
        n_action_steps: int = 10,
        
        # 扩散/流匹配参数
        num_inference_steps: int = 10,
        min_period: float = 0.01,
        max_period: float = 1000.0,
        
        # 时间采样参数
        time_sampling_beta_alpha: float = 1.0,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 1.0,
        time_sampling_offset: float = 0.0,
        
        # 4. 训练策略参数
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        gradient_checkpointing: bool = False,
        dtype: str = "bfloat16",
        
        # 5. 特征描述 (默认为 None，在 init 里初始化)
        input_features: Optional[Dict[str, Any]] = None,
        output_features: Optional[Dict[str, Any]] = None,
        normalization_mapping: Optional[Dict[str, str]] = None,
        
        # 6. RTC 配置
        rtc_config: Optional[Dict[str, Any]] = None,
        
        # 7. 编译选项
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        
        **kwargs
    ):
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.image_resolution = image_resolution
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.min_period = min_period
        self.max_period = max_period
        self.time_sampling_beta_alpha = time_sampling_beta_alpha
        self.time_sampling_beta_beta = time_sampling_beta_beta
        self.time_sampling_scale = time_sampling_scale
        self.time_sampling_offset = time_sampling_offset
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype
        self.compile_model = compile_model
        self.compile_mode = compile_mode

        # 处理 Mutable Defaults (字典不能做默认参数，必须在 init 里赋值)
        if input_features is None:
            self.input_features = {
                "observation.images.cam_high": {"shape": (3, 224, 224), "dtype": "float32"},
                "observation.state": {"shape": (max_state_dim,), "dtype": "float32"},
                "observation.language_instruction": {"shape": (1,), "dtype": "string"}
            }
        else:
            self.input_features = input_features

        if output_features is None:
            self.output_features = {
                "action": {"shape": (max_action_dim,), "dtype": "float32"}
            }
        else:
            self.output_features = output_features

        # 👇 关键修复：确保 normalization_mapping 一定被赋值
        if normalization_mapping is None:
            self.normalization_mapping = {
                "observation.images.cam_high": "identity",
                "observation.state": "mean_std",
                "action": "mean_std",
            }
        else:
            self.normalization_mapping = normalization_mapping

        self.rtc_config = rtc_config

        # 动态调整维度
        if "observation.state" in self.input_features:
            self.max_state_dim = self.input_features["observation.state"]["shape"][0]
        if "action" in self.output_features:
            self.max_action_dim = self.output_features["action"]["shape"][0]

        super().__init__(**kwargs)