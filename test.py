import torch
import os
from pathlib import Path
from transformers import AutoTokenizer

# 引入我们刚才提取的模型类
from model.modeling_pi0 import PI0Policy

# ================= 配置区域 =================
# 请修改为您训练好的模型路径 (包含 config.json, model.safetensors 的文件夹)
# 例如: "outputs/train/pi0_test" 或者您的预训练权重目录
MODEL_PATH = "/root/Users/wangbo/pi0"  
# 如果您的分词器在另一个目录，请修改这里；如果在同一个目录，保持 MODEL_PATH 即可
TOKENIZER_PATH = "/root/Users/wangbo/lerobot/tokenizers/paligemma" 
# ===========================================

def main():
    print(f"🚀 开始加载模型，路径: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  运行设备: {device}")

    # 1. 加载模型 (测试提取的 from_pretrained 逻辑)
    try:
        policy = PI0Policy.from_pretrained(MODEL_PATH)
        policy.to(device)
        policy.eval() # 切换到推理模式
        print("✅ 模型加载成功！")
        
        # 打印一下关键配置，确认 config.py 工作正常
        print(f"   - State Dim: {policy.config.max_state_dim}")
        print(f"   - Action Dim: {policy.config.max_action_dim}")
        print(f"   - Image Size: {policy.config.image_resolution}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 加载分词器 (Tokenizer)
    # Pi0 必须要有分词器才能处理文本指令
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print("✅ 分词器加载成功！")
    except Exception as e:
        print(f"❌ 分词器加载失败 (请检查路径): {e}")
        return

    # 3. 构造伪造输入数据 (Dummy Data)
    print("\n📦 正在构造测试数据...")
    
    # [A] 文本指令
    instruction = "Pick up the blue cube"
    tokenized = tokenizer(
        instruction, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=48, # 这里的长度通常在 config 里，这里暂时写死测试
        truncation=True
    )
    
    # [B] 图像数据 (B, C, H, W)
    # 模拟一张随机噪点的图片
    dummy_image = torch.rand(1, 3, 224, 224).to(device)
    
    # [C] 机械臂状态 (B, State_Dim)
    # 模拟当前机械臂位置
    state_dim = policy.config.max_state_dim
    dummy_state = torch.randn(1, state_dim).to(device)

    # [D] 组装成 Batch 字典
    #这是 modeling_pi0.py 里 forward/select_action 期待的格式
    batch = {
        # 图像 Key (保持不变)
        "observation.images.base_0_rgb": dummy_image, 
        
        # 状态 Key (保持不变)
        "observation.state": dummy_state,
        
        # 🔴 修改这里：改回报错信息里要求的“长名字”
        "observation.language_instruction.input_ids": tokenized["input_ids"].to(device),
        "observation.language_instruction.attention_mask": tokenized["attention_mask"].to(device),
    }

    # 4. 执行推理
    print("⚡ 开始执行推理 (select_action)...")
    try:
        with torch.no_grad():
            # select_action 会返回下一步动作
            action = policy.select_action(batch)
        
        print("\n🎉 推理成功！")
        print(f"📝 输入指令: '{instruction}'")
        print(f"🤖 输出动作形状: {action.shape}")
        print(f"📊 输出动作数值 (前5位): {action[0, :5].cpu().numpy()}...")
        
        # 验证形状是否符合预期: [Action_Dim]
        expected_dim = policy.config.max_action_dim
        if action.shape[-1] == expected_dim:
            print("✅ 输出维度验证通过。")
        else:
            print(f"⚠️  警告: 输出维度 {action.shape[-1]} 与配置 {expected_dim} 不一致")

    except Exception as e:
        print(f"❌ 推理过程中崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()