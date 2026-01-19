import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
import os
from pathlib import Path
from tqdm import tqdm
import logging
import json

# --- 导入自定义模块 ---
from model.config import PI0Config 
from model.modeling_pi0 import PI0Policy
from data.dataset import Pi0Dataset
from utils.normalization import Normalizer

from pathlib import Path
current_dir = Path(__file__).parent
print(current_dir)

# ================= 🔧 训练配置区域 =================
PRETRAINED_MODEL_PATH = "/root/Users/wangbo/pi0"
TOKENIZER_PATH = "/root/Users/wangbo/lerobot/tokenizers/paligemma"
DATASET_ROOT = "/root/Users/wangbo/my_converted_dataset"
STATS_PATH = "/root/Users/wangbo/my_converted_dataset/meta/stats.json" 
OUTPUT_DIR = "/root/Users/wangbo/minipi0/outputs/test"

BATCH_SIZE = 4       
LEARNING_RATE = 1e-4 
NUM_EPOCHS = 10      
SAVE_STEPS = 50      
LOG_STEPS = 1        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================

# def save_checkpoint(model, tokenizer, config, output_dir):
#     """手动保存检查点的辅助函数"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 1. 保存权重 (safetensors)
#     from safetensors.torch import save_file
#     save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
    
#     # 2. 保存 Config
#     config.save_pretrained(output_dir)
    
#     # 3. 保存 Tokenizer
#     tokenizer.save_pretrained(output_dir)
#     print(f"\n💾 手动保存模型到: {output_dir}")

def save_checkpoint(model, tokenizer, config, output_dir):
    """手动保存检查点的辅助函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 获取模型状态字典
    state_dict = model.state_dict()
    
    # -----------------------------------------------------------
    # 🔧 修复 safetensors 共享内存报错 (Weight Tying Fix)
    # -----------------------------------------------------------
    # 报错显示的两个 key 共享了内存，我们需要把其中一个 clone 成独立的
    problematic_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
    
    if problematic_key in state_dict:
        # .clone() 会创建一份新的数据副本，打破内存共享
        state_dict[problematic_key] = state_dict[problematic_key].clone()
    # -----------------------------------------------------------

    # 2. 保存权重 (safetensors)
    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    
    # 3. 保存 Config
    config.save_pretrained(output_dir)
    
    # 4. 保存 Tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"\n💾 手动保存模型到: {output_dir}")

def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 开始训练! 设备: {DEVICE}")

    # 1. 加载统计数据 (CPU)
    print(f"Loading stats from {STATS_PATH}...")
    normalizer = Normalizer(STATS_PATH, device="cpu")
    
    if 'action' in normalizer.stats:
        action_dim = normalizer.stats['action']['mean'].shape[0]
        state_dim = normalizer.stats['observation.state']['mean'].shape[0]
        print(f"📏 检测到数据维度: Action={action_dim}, State={state_dim}")
    else:
        raise ValueError("统计文件中缺少 'action' 字段！")

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 3. 准备数据集
    print("📚 加载数据集...")
    train_dataset = Pi0Dataset(
        root_dir=DATASET_ROOT,
        tokenizer=tokenizer,
        normalizer=normalizer,
        split="train",
        image_size=224,
        action_chunk_size=50
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  
        drop_last=True
    )
    print(f"✅ 数据加载完成，共 {len(train_dataset)} 个样本")

    # 4. 加载并改造模型
    print("🧠 加载模型中...")
    config = PI0Config.from_pretrained(PRETRAINED_MODEL_PATH)
    
    print(f"🔄 修改模型配置: Action Dim {config.max_action_dim} -> {action_dim}")
    config.max_state_dim = state_dim
    config.max_action_dim = action_dim
    
    # 使用我们手写的兼容版 from_pretrained
    policy = PI0Policy.from_pretrained(
        PRETRAINED_MODEL_PATH, 
        config=config, 
        ignore_mismatched_sizes=True
    )
    policy.to(DEVICE)
    policy.train()

    # 5. 优化器
    optimizer = AdamW(policy.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(50, num_training_steps // 10), 
        num_training_steps=num_training_steps
    )

    # 6. 训练循环
    print("🔥 开始循环微调...")
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)
            
            # --- 前向传播 ---
            # 因为我们的 forward 直接返回 loss，不需要判断字典了
            loss = policy(batch) 
            
            # --- 反向传播 ---
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % LOG_STEPS == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # --- 保存 ---
            if global_step % SAVE_STEPS == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                # 使用辅助函数手动保存
                save_checkpoint(policy, tokenizer, config, save_path)
                
    # 7. 最终保存
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    save_checkpoint(policy, tokenizer, config, final_path)
    print(f"\n🎉 训练结束！最终模型保存在: {final_path}")

if __name__ == "__main__":
    main()