import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
from pathlib import Path

# --- 导入项目模块 ---
try:
    from model.modeling_pi0 import PI0Policy
    from utils.normalization import Normalizer
    from data.dataset import Pi0Dataset
    print("✅ 成功导入所有自定义模块 (model, utils, data)")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# ================= 🔧 配置区域 =================
# 1. 模型路径
MODEL_PATH = "/root/Users/wangbo/pi0"
# 2. 分词器路径
TOKENIZER_PATH = "/root/Users/wangbo/lerobot/tokenizers/paligemma"
# 3. 数据集路径
DATASET_ROOT = "/root/Users/wangbo/my_converted_dataset_v3"
# ==============================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  运行设备: {device}")

    # ---------------------------------------------------------
    # 1. 准备组件
    # ---------------------------------------------------------
    print("\n[Step 1] 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print("✅ Tokenizer 就绪")
    except Exception as e:
        print(f"❌ Tokenizer 失败: {e}")
        return

    # ---------------------------------------------------------
    # 2. 加载数据集 (关键修改：强制 normalizer=None)
    # ---------------------------------------------------------
    print(f"\n[Step 2] 加载数据集: {DATASET_ROOT}")
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 错误: 路径不存在 {DATASET_ROOT}")
        return

    try:
        # ⚠️ 关键点：normalizer=None
        # 防止 7维真实数据 撞上 32维假统计文件 导致报错
        dataset = Pi0Dataset(
            root_dir=DATASET_ROOT,
            tokenizer=tokenizer,
            normalizer=None,  # <--- 必须是 None
            split="train",
            image_size=224,
            action_chunk_size=50
        )
        print(f"✅ Dataset 初始化成功!")
        print(f"   - 根目录: {dataset.root_dir}")
        print(f"   - 总帧数: {len(dataset)}")
        
        # 读取一个 Batch
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        print("🚚 正在从 Parquet 读取 Batch (包含视频解码)...")
        
        batch = next(iter(loader))
        
        # 转移到 GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        print("✅ 数据读取成功! 形状检查:")
        print(f"   - Images: {batch['observation.images.base_0_rgb'].shape}")
        # 这里应该显示 [2, 7] (真实维度)
        print(f"   - State:  {batch['observation.state'].shape}") 
        # 这里应该显示 [2, 50, 7] (真实维度)
        print(f"   - Action: {batch['action'].shape}") 

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. 加载模型 & 联合推理
    # ---------------------------------------------------------
    print("\n[Step 3] 模型联合推理...")
    try:
        policy = PI0Policy.from_pretrained(MODEL_PATH)
        policy.to(device)
        policy.eval()
        
        # ⚠️ 这里的逻辑很重要：
        # Base模型配置是 32维，数据是 7维。
        # 我在 modeling_pi0.py 里写的 prepare_state 会自动进行 Padding (补零)，
        # 所以直接传进去应该不会报错，会自动补齐到 32维。
        
        print("⚡ 开始执行 select_action...")
        with torch.no_grad():
            raw_actions = policy.select_action(batch)
            
        print("🎉🎉🎉 全流程跑通！")
        print(f"🤖 模型输出动作形状: {raw_actions.shape}")
        
        if raw_actions.shape[-1] == 32:
             print("💡 验证通过：模型成功接受了 7维数据(自动补零)，并输出了 32维结果(Base默认)。")

    except Exception as e:
        print(f"❌ 推理崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()