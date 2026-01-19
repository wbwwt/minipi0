import torch
import numpy as np
import os
from transformers import AutoTokenizer
from pathlib import Path

# --- 导入自定义模块 ---
from model.config import PI0Config 
from model.modeling_pi0 import PI0Policy
from data.dataset import Pi0Dataset
from utils.normalization import Normalizer

# ================= 🔧 推理配置区域 =================
# 1. 模型路径
TRAINED_MODEL_PATH = "/root/Users/wangbo/minipi0/outputs/test/checkpoint-900"

# 2. 基础配置
TOKENIZER_PATH = "/root/Users/wangbo/lerobot/tokenizers/paligemma" 
# 如果本地没有，可以用 google 官方的: "google/paligemma-3b-pt-224"
if not os.path.exists(TOKENIZER_PATH):
    TOKENIZER_PATH = "google/paligemma-3b-pt-224"

DATASET_ROOT = "/root/Users/wangbo/my_converted_dataset"
STATS_PATH = "/root/Users/wangbo/my_converted_dataset/meta/stats.json"

# 3. 硬件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================

def main():
    print(f"🚀 开始推理评估! 设备: {DEVICE}")

    # ---------------------------------------------------------
    # 1. 加载统计数据 (必须在加载模型前获取真实维度)
    # ---------------------------------------------------------
    print(f"Loading stats from {STATS_PATH}...")
    # ⚠️ 必须用 device="cpu"，方便后续 numpy 转换
    normalizer = Normalizer(STATS_PATH, device="cpu") 
    
    if 'action' in normalizer.stats:
        real_action_dim = normalizer.stats['action']['mean'].shape[0]
        real_state_dim = normalizer.stats['observation.state']['mean'].shape[0]
        print(f"📏 真实数据维度: Action={real_action_dim}, State={real_state_dim}")
    else:
        raise ValueError("统计文件中缺少 'action' 字段！")

    # ---------------------------------------------------------
    # 2. 加载模型与配置
    # ---------------------------------------------------------
    print(f"🧠 Loading trained model from: {TRAINED_MODEL_PATH}")
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"❌ 错误: 找不到模型路径 {TRAINED_MODEL_PATH}")
        return

    # A. 加载 Config
    config = PI0Config.from_pretrained(TRAINED_MODEL_PATH)
    
    # 🔧【关键修复】强制修正 Config 里的维度设置
    # 因为底座默认可能是 32 维，而您的数据是 7 维
    # 如果不修正，加载权重时会报错 size mismatch
    if config.max_action_dim != real_action_dim:
        print(f"⚠️ 发现维度不匹配 (Config={config.max_action_dim} vs Stats={real_action_dim})")
        print(f"🔧 正在强制修正 Config 以匹配您的数据...")
        config.max_action_dim = real_action_dim
        config.max_state_dim = real_state_dim

    # B. 加载权重
    # ignore_mismatched_sizes=True 是为了防止一些无关紧要的头信息报错
    policy = PI0Policy.from_pretrained(
        TRAINED_MODEL_PATH, 
        config=config,
        ignore_mismatched_sizes=True 
    )
    policy.to(DEVICE)
    policy.eval() 

    # C. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # ---------------------------------------------------------
    # 3. 准备测试样本
    # ---------------------------------------------------------
    # 使用 train split 确保一定能读到数据 (即使只有一个 episode)
    dataset = Pi0Dataset(
        root_dir=DATASET_ROOT,
        tokenizer=tokenizer,
        normalizer=normalizer,
        split="train", 
        image_size=224,
        action_chunk_size=50
    )
    
    # 随机取一个样本 (例如第 50 帧)
    sample_idx = 0
    if len(dataset) > 50: sample_idx = 50
    
    print(f"🧪 抽取第 {sample_idx} 帧作为测试样本...")
    sample = dataset[sample_idx]

    # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(DEVICE) 

    # ---------------------------------------------------------
    # 4. 执行推理 (生成完整轨迹)
    # ---------------------------------------------------------
    print("⚡ 模型正在预测完整动作轨迹 (Chunk)...")
    
    with torch.no_grad():
        # ❌ 不使用 select_action (它只返回第 1 步)
        # ✅ 使用 sample_actions (返回未来 50 步)
        
        # 1. 预处理
        images, img_masks = policy._preprocess_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language_instruction.input_ids"]
        lang_masks = batch["observation.language_instruction.attention_mask"]

        # 2. 生成轨迹 [Batch, Chunk, Dim]
        full_actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state
        )

        # 3. 截断到有效维度 (防止 padding 干扰)
        pred_actions_normalized = full_actions[:, :, :real_action_dim]

    # ---------------------------------------------------------
    # 5. 后处理与展示
    # ---------------------------------------------------------
    # 搬回 CPU
    pred_actions_normalized = pred_actions_normalized.cpu()
    gt_actions_normalized = sample["action"].unsqueeze(0).cpu()

    # 检查维度
    if pred_actions_normalized.shape[-1] != real_action_dim:
        print(f"❌ 维度依然错误: {pred_actions_normalized.shape}")
        return

    # 反归一化
    print("🔄 正在反归一化 (还原为真实物理数值)...")
    pred_actions_real = normalizer.denormalize(pred_actions_normalized, key="action")
    gt_actions_real = normalizer.denormalize(gt_actions_normalized, key="action")

    # 打印表格
    print("\n" + "="*65)
    print(f"📊 动作轨迹对比 (Action Chunking, 第 1 个关节)")
    print("="*65)
    
    pred_np = pred_actions_real[0].numpy() # [Chunk, Dim]
    gt_np = gt_actions_real[0].numpy()     # [Chunk, Dim]
    
    print(f"{'Step (未来)':<12} | {'预测值 (Pred)':<15} | {'真实值 (GT)':<15} | {'误差 (Diff)':<15}")
    print("-" * 65)
    
    # 打印前 10 步，看看连贯性
    for t in range(10): 
        val_pred = pred_np[t, 0] 
        val_gt = gt_np[t, 0]
        diff = abs(val_pred - val_gt)
        print(f"T + {t:<8} | {val_pred:<15.4f} | {val_gt:<15.4f} | {diff:<15.4f}")

    # 计算整体 Chunk 的误差
    mse = np.mean((pred_np - gt_np) ** 2)
    print("-" * 65)
    print(f"📉 整个轨迹 (50步) 的均方误差 (MSE): {mse:.6f}")
    print("=" * 65)

    if mse < 0.1:
        print("✅ 成功！模型不仅预测了当前动作，还规划了未来轨迹。")
    else:
        print("⚠️ 误差较大，可能模型还未收敛。")

if __name__ == "__main__":
    main()