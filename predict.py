import sys
import os
import time
import torch
import numpy as np
import cv2
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

# ==================== 0. 路径与依赖配置 ====================
# 必须先执行这一步，确保 Python 能找到 startouch 驱动
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTOUCH_PATH = os.path.join(CURRENT_DIR, 'startouch-v1', 'interface_py')

if STARTOUCH_PATH not in sys.path:
    print(f"🔧 将驱动路径加入系统 Path: {STARTOUCH_PATH}")
    sys.path.append(STARTOUCH_PATH)

# 导入项目模块
from utils.camera import RealSenseCamera
from model.config import PI0Config
from model.modeling_pi0 import PI0Policy
from utils.normalization import Normalizer

# ==================== 🔧 配置区域 ====================
# 模型路径
MODEL_PATH = "/home/lumos/pi0/weight/checkpoint-28000"
STATS_PATH = "/home/lumos/pi0/replay_remote_ctrl/my_converted_dataset_v3/meta/stats.json"
TOKENIZER_PATH = "/home/lumos/pi0/replay_remote_ctrl/minipi0/paligemma" # 请修改

# 任务描述
TASK_DESC = "pick up the cube"

# 硬件配置
CAN_INTERFACE = "can0"     # Startouch 默认 CAN 口
CONTROL_HZ = 30            # 控制频率
DT = 1.0 / CONTROL_HZ
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================

def setup_robot(can_interface="can0", enable_gripper=True):
    """
    初始化 Startouch 机械臂
    """
    print(f"🤖 正在初始化 Startouch 机械臂 (CAN: {can_interface})...")
    
    try:
        # 延迟导入，防止因缺少 .so 文件导致脚本直接崩溃无法捕获异常
        sys.path.append('/home/lumos/pi0/replay_remote_ctrl/startouch-v1/interface_py')
        # from startouchclass import SingleArm
        from startouchclass import SingleArm
        
        # 初始化机械臂对象
        # 注意：SingleArm 内部会自动加载 param_csv_gripper 下的参数文件
        # 只要保持目录结构不变 (minipi0/startouch-v1/...)，它就能找到
        robot = SingleArm(can_interface_=can_interface, gripper=enable_gripper)
        robot.set_joint([-0.01468681 , 0.58384833 , 0.23212787 , 0.38242924 ,-0.04100862 , 0.        ] , tf=3)

        
        print("✅ Startouch 机械臂连接成功")
        
        # 可选：进行一些自检或归位
        # print("   正在归位 (Home)...")
        # robot.go_home()
        
        return robot
        
    except ImportError as e:
        print(f"❌ 无法导入 Startouch 驱动: {e}")
        print(f"   请检查 {STARTOUCH_PATH} 下是否存在 .so 文件")
        raise e
    except Exception as e:
        print(f"❌ 机械臂初始化失败: {e}")
        # 这里可能需要检查 CAN 卡是否启用: sudo ip link set can0 up type can bitrate 1000000
        raise e

def main():
    print(f"🚀 启动实机推理... 设备: {DEVICE}")

    # ---------------------------------------------------------
    # 1. 硬件初始化 (Robot & Camera)
    # ---------------------------------------------------------
    try:
        robot = setup_robot(can_interface=CAN_INTERFACE, enable_gripper=True)
    except Exception:
        return # 初始化失败直接退出

    print("📷 正在打开 RealSense 摄像头...")
    try:
        camera = RealSenseCamera(width=640, height=480, fps=30)
    except Exception as e:
        print(f"❌ 摄像头打开失败: {e}")
        print("   请确保已连接 USB3.0 接口并安装了 pyrealsense2")
        return

    # ---------------------------------------------------------
    # 2. 模型加载
    # ---------------------------------------------------------
    print("🧠 加载 Pi0 模型...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    config = PI0Config.from_pretrained(MODEL_PATH)
    config.max_action_dim = 8
    config.max_state_dim = 8

    # 🔧 定义 4-bit 量化配置
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16, # 计算时用 bf16
    #     bnb_4bit_quant_type="nf4",             # 使用 nf4 格式精度更高
    #     bnb_4bit_use_double_quant=True         # 二次量化，进一步省显存
    # )

    # 🔧 定义 4-bit 量化配置 (改用 float16)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.float16, # 👈 修改这里：用 float16
        bnb_4bit_quant_type="nf4",             
        bnb_4bit_use_double_quant=True         
    )

    policy = PI0Policy.from_pretrained(
        MODEL_PATH, 
        config=config,
        # torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )
    # policy = PI0Policy.from_pretrained(MODEL_PATH, config=config)
    # policy.to(DEVICE)
    # policy.to(device=DEVICE, dtype=torch.bfloat16)
    policy.eval()

    print(f"📊 加载统计数据: {STATS_PATH}")
    # normalizer = Normalizer(STATS_PATH, device=DEVICE)
    normalizer = Normalizer(STATS_PATH, device='cpu')
    # 预处理文本
    text_tokens = tokenizer(TASK_DESC, return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    input_ids = text_tokens["input_ids"].to(DEVICE)
    attention_mask = text_tokens["attention_mask"].to(DEVICE)

    print("\n✅ 系统就绪! 此时请确保机械臂周围安全。")
    print("👉 按下 Enter 键开始执行推理...")
    input()

    # ---------------------------------------------------------
    # 3. 控制循环
    # ---------------------------------------------------------
    print(f"🔥 开始执行循环 ({CONTROL_HZ} Hz)... 按 Ctrl+C 停止")
    
    try:
        step = 0
        while True:
            t_start = time.time()

            # --- A. 获取传感器数据 ---
            img_bgr = camera.get_frame()
            if img_bgr is None:
                print("⚠️ 丢帧 (Camera)")
                continue
            
            # 获取机械臂状态 [j1...j6, gripper]
            # joints = robot.get_joint_positions()   # np.array (6,)
            # joints = robot.get_ee_pose_quat()
            # gripper = robot.get_gripper_position() # float
            # current_state = np.append(joints, gripper).astype(np.float32)
            
            pos_t, quat_t = robot.get_ee_pose_quat() 
            # pos_t: [x,y,z], quat_t: [w,x,y,z] (根据你的描述)
            
            # 2. 获取夹爪
            gripper = robot.get_gripper_position()
            
            # 3. 拼接 [x,y,z, w,x,y,z, gripper] (共8维)
            # ⚠️ 极其重要: 确认你训练时的四元数顺序是 wxyz 还是 xyzw？
            # 如果训练集是 [x,y,z,qx,qy,qz,qw]，你需要把 quat_t[0] (w) 挪到最后
            # 假设训练集通常是 [x,y,z, qx,qy,qz,qw]，如下调整：
            # quat_reordered = np.array([quat_t[1], quat_t[2], quat_t[3], quat_t[0]]) 
            # 这里先按你的原始输出拼接，请务必核对！
            
            current_state = np.concatenate([pos_t, quat_t, [gripper]]).astype(np.float32)

            # # --- B. 数据预处理 ---
            # # 图像: BGR->RGB, Resize, Normalize
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # if img_rgb.shape[0] != IMAGE_SIZE or img_rgb.shape[1] != IMAGE_SIZE:
            #      img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            # else:
            #      img_resized = img_rgb
                 
            # img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            # img_tensor = img_tensor.unsqueeze(0).to(DEVICE) 

            # # 状态: Normalize
            # state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(DEVICE) 
            # state_norm = normalizer.normalize(state_tensor, key="observation.state")

            # # --- C. 模型推理 ---
            # batch = {
            #     "observation.images.base_0_rgb": img_tensor,
            #     "observation.state": state_norm,
            #     "observation.language_instruction.input_ids": input_ids,
            #     "observation.language_instruction.attention_mask": attention_mask
            # }

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[0] != IMAGE_SIZE or img_rgb.shape[1] != IMAGE_SIZE:
                 img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            else:
                 img_resized = img_rgb
            
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            # 👈 修改这里：用 float16
            # img_tensor = img_tensor.unsqueeze(0).to(DEVICE, dtype=torch.float16)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            # 状态处理 (CPU Float32 计算 -> GPU Float16)
            state_tensor_cpu = torch.from_numpy(current_state).float().unsqueeze(0) 
            
            stats_mean = normalizer.stats["observation.state"]["mean"].cpu()
            stats_std = normalizer.stats["observation.state"]["std"].cpu()
            
            state_norm_cpu = (state_tensor_cpu - stats_mean) / (stats_std + 1e-8)
            
            # 👈 修改这里：用 float16
            # state_norm = state_norm_cpu.to(DEVICE, dtype=torch.float16)
            state_norm = state_norm_cpu.to(DEVICE)

            # --- C. 模型推理 ---
            batch = {
                "observation.images.base_0_rgb": img_tensor,
                "observation.state": state_norm,
                "observation.language_instruction.input_ids": input_ids,
                "observation.language_instruction.attention_mask": attention_mask
            }

            with torch.no_grad():
                # Action Chunking 逻辑封装在 select_action 内
                action_norm = policy.select_action(batch)

            # ***********************************
            action_norm = action_norm.cpu()

            # --- D. 执行动作 ---
            action_real = normalizer.denormalize(action_norm, key="action")
            action_np = action_real.squeeze(0).cpu().numpy()

            # 解析动作
            target_joints = action_np[:-1] # 前7个
            target_gripper = action_np[-1] # 最后1个

            # 发送指令 (透传)
            # 速度设为0，由底层控制器负责插补
            # robot.set_joint_raw(target_joints, velocities=[0.0]*6)
            robot.set_end_effector_pose_quat_raw(target_joints[:3], target_joints[3:])
            robot.setGripperPosition_raw(target_gripper)

            # --- E. 频率控制 ---
            elapsed = time.time() - t_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 状态监控了 bnb_4bit_compute_dtype=torch.floa
            if step % 30 == 0:
                print(f"Step {step} | Gripper: {gripper:.2f}->{target_gripper:.2f}")
            step += 1

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("资源清理中...")
        if 'camera' in locals(): camera.stop()
        # robot 对象通常不需要显式 close，析构时会自动释放
        # 如果需要回零: robot.go_home()

if __name__ == "__main__":
    main()