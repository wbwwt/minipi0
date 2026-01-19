import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

# 1. 导入您的自定义模块
try:
    from model.modeling_pi0 import PI0Policy
    print("✅ 成功导入 model.modeling_pi0")
except ImportError as e:
    print(f"❌ 导入 model 失败: {e}")
    exit(1)

try:
    from utils.normalization import Normalizer
    print("✅ 成功导入 utils.normalization")
except ImportError as e:
    print(f"❌ 导入 utils 失败: {e}")
    exit(1)

# ================= 配置路径 =================
# 请根据您实际的权重路径修改这里
MODEL_PATH = "/root/Users/wangbo/pi0"
# 通常 stats 文件在模型目录下，叫 dataset_stats.json 或 stats.json
STATS_PATH = f"{MODEL_PATH}/dataset_stats.json" 

TOKENIZER_PATH = "/root/Users/wangbo/lerobot/tokenizers/paligemma"
# ===========================================

def main():
    print(f"\n🚀 开始全系统验证...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  运行设备: {device}")

    # ---------------------------------------------------------
    # 步骤 1: 验证 Normalizer (utils/normalization.py)
    # ---------------------------------------------------------
    print("\n[Step 1] 验证归一化模块...")
    normalizer = None
    if os.path.exists(STATS_PATH):
        try:
            normalizer = Normalizer(STATS_PATH, device=device)
            print(f"✅ Normalizer 加载成功! (读取自 {STATS_PATH})")
            # 简单检查内容
            if "action" in normalizer.stats:
                print(f"   - 包含 Action 统计: Mean shape {normalizer.stats['action']['mean'].shape}")
            else:
                print("⚠️  警告: 统计文件中没有 'action' 字段")
        except Exception as e:
            print(f"❌ Normalizer 初始化出错: {e}")
            return
    else:
        print(f"⚠️  警告: 未找到统计文件 {STATS_PATH}，将跳过数值还原测试。")

    # ---------------------------------------------------------
    # 步骤 2: 验证模型加载 (model/modeling_pi0.py)
    # ---------------------------------------------------------
    print("\n[Step 2] 验证模型加载...")
    try:
        policy = PI0Policy.from_pretrained(MODEL_PATH)
        policy.to(device)
        policy.eval()
        print("✅ Pi0 模型加载成功!")
        print(f"   - Config State Dim: {policy.config.max_state_dim}")
        print(f"   - Config Action Dim: {policy.config.max_action_dim}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # ---------------------------------------------------------
    # 步骤 3: 准备 Tokenizer (HuggingFace)
    # ---------------------------------------------------------
    print("\n[Step 3] 准备 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH) # 通常在同一目录
        print("✅ Tokenizer 加载成功!")
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        return

    # ---------------------------------------------------------
    # 步骤 4: 构造数据并推理 (Integration Test)
    # ---------------------------------------------------------
    print("\n[Step 4] 执行完整推理流程...")
    
    # 4.1 构造文本
    text = "Pick up the apple"
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    
    # 4.2 构造 Dummy 图像和状态
    # 注意：这里使用您之前报错信息里确认的 Key 名称
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_state = torch.randn(1, policy.config.max_state_dim).to(device)
    
    # 4.3 归一化输入状态 (如果有 Normalizer)
    if normalizer:
        # 模拟：假设 dummy_state 是真实世界的数值，我们需要先把它归一化再喂给模型
        # 公式: (real - mean) / std
        model_input_state = normalizer.normalize(dummy_state, key="observation.state")
    else:
        model_input_state = dummy_state

    # 4.4 组装 Batch
    batch = {
        "observation.images.base_0_rgb": dummy_image,
        "observation.state": model_input_state,
        "observation.language_instruction.input_ids": tokens["input_ids"].to(device),
        "observation.language_instruction.attention_mask": tokens["attention_mask"].to(device),
    }

    # 4.5 模型推理
    try:
        with torch.no_grad():
            raw_action = policy.select_action(batch)
        print("✅ 模型推理成功! 获得原始输出 (Normalized Action)。")
    except Exception as e:
        print(f"❌ 推理过程崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 步骤 5: 反归一化验证 (Output Verification)
    # ---------------------------------------------------------
    print("\n[Step 5] 验证结果反归一化...")
    
    print(f"🤖 原始输出 (前4位): {raw_action[0, :4].cpu().numpy()}")
    
    if normalizer:
        try:
            # 核心测试：把模型输出还原为真实物理量
            real_action = normalizer.denormalize(raw_action, key="action", mode="mean_std")
            
            print(f"🦾 真实动作 (前4位): {real_action[0, :4].cpu().numpy()}")
            print("\n🎉🎉🎉 验证通过！所有模块工作正常！")
            print("您的 Mini-Pi0 项目现在可以独立运行了。")
            
        except Exception as e:
            print(f"❌ 反归一化计算失败: {e}")
    else:
        print("⚠️  跳过反归一化（因为没有加载 Normalizer）")

if __name__ == "__main__":
    main()