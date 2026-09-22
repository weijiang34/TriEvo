import numpy as np


def calc_params(
    batch_size,
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    d_ff=None,
):
    is_swiglu = True  # 假设使用 SwiGLU 激活函数
    if d_ff is None:
        # SwiGLU 通常将隐藏层维度缩减为 8/3 d_model
        d_ff = int(8 / 3 * d_model) if is_swiglu else 4 * d_model

    # 1. 单层 Transformer 参数量 (忽略微小的 LayerNorm 偏置)
    attn_params = 4 * d_model * d_model
    ffn_params = (
        3 * d_ff * d_model if is_swiglu else 2 * d_ff * d_model
    )
    model_params = num_layers * (attn_params + ffn_params)

    # 2. 嵌入层与输出层 (不共享权重)
    model_params += 2 * vocab_size * d_model

    # 3. 训练状态下的静态元素量 (混合精度 AdamW)
    # 参数(FP16) = 2字节, 梯度(FP16) = 2字节, 优化器(FP32) = 12字节 -> 共 16 字节
    # 这里我们只计算逻辑上的“参数等效数量”
    
    # 4. 激活值元素量估算 (前向传播保留用于反向传播的张量)
    # 每一层：LN(2) + QKV_proj(3) + Attn_rotary/mask(1) + Attn_out(2) + FFN(3或4)
    # 以下为标准 Transformer 激活值元素粗略估算
    b, s, h = batch_size, context_length, d_model
    
    # 单层激活值
    layer_act = (
        2 * b * s * h  # Input LN
        + 3 * b * s * h  # QKV Proj
        + b * num_heads * s * s  # Attention Matrix
        + 2 * b * s * h  # Attn Out Proj
        + (4 * b * s * d_ff if is_swiglu else 2 * b * s * d_ff)  # FFN
    )
    total_activations = num_layers * layer_act + 2 * b * s * h

    # 5. 显存计算 (单位: GB)
    # 参数 2字节 + 梯度 2字节 + 优化器 12字节 = 16字节
    static_mem_gb = (model_params * (2 + 2 + 12)) / (1024**3)
    # 激活值以 FP16 存储 = 2字节
    activation_mem_gb = (total_activations * 2) / (1024**3)

    total_memory_gb = static_mem_gb + activation_mem_gb

    return {
        "model_params_billions": model_params / 1e9,
        "static_mem_gb": static_mem_gb,
        "activation_mem_gb": activation_mem_gb,
        "total_memory_gb": total_memory_gb,
    }


def dataset_token_count(path) -> int:
    return len(np.memmap(path, dtype="uint16", mode="r"))
