import triton
import triton.language as tl
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.cuda.nvtx as nvtx

# autotune_configs = [
#     # 根据硬件自动测试不同的 BLOCK 组合和流水线级数 (num_stages)
#     triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=4),
#     triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_stages=4, num_warps=8),
#     triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=4),
# ]

# @triton.autotune(autotune_configs, key=['N_QUERIES', 'N_KEYS', 'D'])
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_L: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    m_prev = tl.full((Q_TILE_SIZE, 1), -float('inf'), dtype=tl.float32)
    l_prev = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    o_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    compute_dtype = Q_block.dtype
    
    for j in tl.range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load current block of K and V
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        S_block = tl.dot(Q_block.to(compute_dtype), tl.trans(K_block.to(compute_dtype))) * scale   # (Q_TILE_SIZE, K_TILE_SIZE)
        
        if IS_CAUSAL:
            qstart = query_tile_index * Q_TILE_SIZE
            kstart = j * K_TILE_SIZE
            q_indices = tl.arange(0, Q_TILE_SIZE) + qstart
            k_indices = tl.arange(0, K_TILE_SIZE) + kstart
            mask = q_indices[:, None] < k_indices[None, :]
            S_block = tl.where(mask, float("-inf"), S_block)

        m_new = tl.maximum(m_prev, tl.max(S_block, axis=1, keep_dims=True)) # (Q_TILE_SIZE, 1)

        P = tl.exp(S_block - m_new)    # (Q_TILE_SIZE, K_TILE_SIZE) - (Q_TILE_SIZE, 1) = (Q_TILE_SIZE, K_TILE_SIZE)

        l_new = tl.exp(m_prev - m_new) * l_prev + tl.sum(P, axis=1, keep_dims=True)     # (Q_TILE_SIZE, 1) * (Q_TILE_SIZE, 1) + (Q_TILE_SIZE, 1) = (Q_TILE_SIZE, 1)
        o_new = tl.exp(m_prev - m_new) * o_prev + tl.dot(P.to(compute_dtype), V_block.to(compute_dtype))    # (Q_TILE_SIZE, D)
                # (Q_TILE_SIZE, 1) * (Q_TILE_SIZE, D) + (Q_TILE_SIZE, K_TILE_SIZE) @ (K_TILE_SIZE, D) = (Q_TILE_SIZE, D)
        m_prev = m_new
        l_prev = l_new
        o_prev = o_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = o_prev / l_prev    # (Q_TILE_SIZE, D) / (Q_TILE_SIZE, 1) = (Q_TILE_SIZE, D)
    tl.store(O_block_ptr, output.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    if STORE_L:
        L = m_prev + tl.log(l_prev)
        tl.store(L_block_ptr, tl.ravel(L), boundary_check=(0,))


@triton.jit
def flash_decode_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
    stride_qb, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_od,
    CONTEXT_LENGTH,
    scale,
    D: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Attend one query against one contiguous KV-cache sequence per program."""
    batch_head_index = tl.program_id(0)
    offsets_d = tl.arange(0, D)
    query = tl.load(Q_ptr + batch_head_index * stride_qb + offsets_d * stride_qd)

    max_score = -float("inf")
    score_sum = 0.0
    output = tl.zeros((D,), dtype=tl.float32)

    for key_start in tl.range(0, CONTEXT_LENGTH, K_TILE_SIZE):
        offsets_k = key_start + tl.arange(0, K_TILE_SIZE)
        key_mask = offsets_k < CONTEXT_LENGTH
        cache_offsets = (
            batch_head_index * stride_kb
            + offsets_k[:, None] * stride_kk
            + offsets_d[None, :] * stride_kd
        )
        key_block = tl.load(
            K_cache_ptr + cache_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )
        value_offsets = (
            batch_head_index * stride_vb
            + offsets_k[:, None] * stride_vk
            + offsets_d[None, :] * stride_vd
        )
        value_block = tl.load(
            V_cache_ptr + value_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )

        scores = tl.sum(key_block * query[None, :], axis=1) * scale
        scores = tl.where(key_mask, scores, -float("inf"))
        next_max_score = tl.maximum(max_score, tl.max(scores, axis=0))
        probabilities = tl.exp(scores - next_max_score)
        correction = tl.exp(max_score - next_max_score)
        score_sum = correction * score_sum + tl.sum(probabilities, axis=0)
        output = correction * output + tl.sum(
            value_block * probabilities[:, None], axis=0
        )
        max_score = next_max_score

    tl.store(
        O_ptr + batch_head_index * stride_ob + offsets_d * stride_od,
        (output / score_sum).to(O_ptr.dtype.element_ty),
    )

# @triton.autotune(autotune_configs, key=['N_QUERIES', 'N_KEYS', 'D'])
@triton.jit
def flash_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    D_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq, stride_dd,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,    # d_model
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # Program indices
    k_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq, stride_dd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
    V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

    dK_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # load blocks
        Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
        dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
        D_block = tl.load(D_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, 1)
        L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)

        compute_dtype = Q_block.dtype

        # recompute S and P
        S_block = tl.dot(Q_block.to(compute_dtype), tl.trans(K_block.to(compute_dtype))) * scale
        if IS_CAUSAL:
            q_start = i * Q_TILE_SIZE
            k_start = k_tile_index * K_TILE_SIZE
            q_indices = tl.arange(0, Q_TILE_SIZE) + q_start
            k_indices = tl.arange(0, K_TILE_SIZE) + k_start
            mask = q_indices[:, None] < k_indices[None, :]
            S_block = tl.where(mask, float("-inf"), S_block)
        P_block = tl.exp(S_block - L_block[:,None])

        # compute gradients
        dV_block = dV_block + tl.dot(tl.trans(P_block.to(compute_dtype)), dO_block.to(compute_dtype))
        dP_block = tl.dot(dO_block.to(compute_dtype), tl.trans(V_block.to(compute_dtype)))
        dS_block = P_block * (dP_block - D_block)
        dK_block = dK_block + tl.dot(tl.trans(dS_block.to(compute_dtype)), Q_block.to(compute_dtype)) * scale

        # advance blocks
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_block.to(dK_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_block.to(dV_ptr.dtype.element_ty), boundary_check=(0, 1))

# @triton.autotune(autotune_configs, key=['N_QUERIES', 'N_KEYS', 'D'])
@ triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    D_ptr, L_ptr,
    dQ_ptr, 
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq, stride_dd,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,    # d_model
    Q_TILE_SIZE: tl.constexpr,  # d_q
    K_TILE_SIZE: tl.constexpr,  # d_k
    IS_CAUSAL: tl.constexpr
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer with corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq, stride_dd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    D_block = tl.load(D_block_ptr, boundary_check=(0, 1), padding_option="zero")  #(Q_TILE_SIZE, 1)
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)

    compute_dtype = Q_block.dtype

    dQ_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        
        # load blocks
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        # recompute S and P
        S_block = tl.dot(Q_block.to(compute_dtype), tl.trans(K_block.to(compute_dtype))) * scale
        if IS_CAUSAL:
            q_start = query_tile_index * Q_TILE_SIZE
            k_start = j * K_TILE_SIZE
            q_indices = tl.arange(0, Q_TILE_SIZE) + q_start
            k_indices = tl.arange(0, K_TILE_SIZE) + k_start
            mask = q_indices[:, None] < k_indices[None, :]
            S_block = tl.where(mask, float("-inf"), S_block)
        P_block = tl.exp(S_block - L_block[:,None])

        # compute gradients
        dP_block = tl.dot(dO_block.to(compute_dtype), tl.trans(V_block.to(compute_dtype)))
        dS_block = P_block * (dP_block - D_block)
        dQ_block = dQ_block + tl.dot(dS_block.to(compute_dtype), K_block.to(compute_dtype)) * scale

        # advance blocks
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_block.to(dQ_ptr.dtype.element_ty), boundary_check=(0, 1))

class FA2Triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, Q, K, V, is_causal):
        with nvtx.range("forward.FA2Triton"):
            with nvtx.range("forward.FA2Triton.allocate"):
                B, L, D = Q.shape
                assert K.shape == (B, L, D)
                assert V.shape == (B, L, D)

                # Allocate output tensors
                output = torch.empty_like(Q)
                logsumexp = torch.zeros((B, L), device=Q.device, dtype=Q.dtype)

                # Define tile sizes
                ctx.Q_TILE_SIZE = 64
                ctx.K_TILE_SIZE = 64
                ctx.D = D

            with nvtx.range("forward.FA2Triton.kernel"):
                # Launch the Triton kernel
                grid = (L // ctx.Q_TILE_SIZE + (L % ctx.Q_TILE_SIZE != 0), B)
                flash_fwd_kernel[grid](
                    Q, K, V,
                    output, logsumexp,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    K.stride(0), K.stride(1), K.stride(2),
                    V.stride(0), V.stride(1), V.stride(2),
                    output.stride(0), output.stride(1), output.stride(2),
                    logsumexp.stride(0), logsumexp.stride(1),
                    L, L,
                    1 / (ctx.D ** 0.5),
                    D=ctx.D,
                    Q_TILE_SIZE=ctx.Q_TILE_SIZE,
                    K_TILE_SIZE=ctx.K_TILE_SIZE,
                    IS_CAUSAL=is_causal,
                    STORE_L=True,
                )

            with nvtx.range("forward.FA2Triton.save_for_backward"):
                ctx.save_for_backward(Q, K, V, output, logsumexp)
                ctx.is_causal = is_causal

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        with nvtx.range("backward.FA2Triton"):
            with nvtx.range("backward.FA2Triton.allocate"):
                Q, K, V, output, logsumexp = ctx.saved_tensors
                B, L, D = Q.shape
                assert K.shape == (B, L, D)
                assert V.shape == (B, L, D)
                assert grad_output.shape == (B, L, D)

                # pre-compute D
                D_mat = torch.sum(torch.einsum('bqd,bqd->bqd', output, grad_output), dim=-1, keepdim=True)  # (B, L, 1)

                # Allocate gradient tensors
                dQ = torch.zeros_like(Q)
                dK = torch.zeros_like(K)
                dV = torch.zeros_like(V)

            with nvtx.range("backward.FA2Triton.kernel_dk_dv"):
                grid = (L // ctx.K_TILE_SIZE + (L % ctx.K_TILE_SIZE != 0), B)
                flash_bwd_dk_dv_kernel[grid](
                    Q, K, V, grad_output,
                    D_mat, logsumexp,
                    dK, dV,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    K.stride(0), K.stride(1), K.stride(2),
                    V.stride(0), V.stride(1), V.stride(2),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
                    D_mat.stride(0), D_mat.stride(1), D_mat.stride(2),
                    logsumexp.stride(0), logsumexp.stride(1),
                    dK.stride(0), dK.stride(1), dK.stride(2),
                    dV.stride(0), dV.stride(1), dV.stride(2),
                    L, L,
                    1 / (ctx.D ** 0.5),
                    D=ctx.D,
                    Q_TILE_SIZE=ctx.Q_TILE_SIZE,
                    K_TILE_SIZE=ctx.K_TILE_SIZE,
                    IS_CAUSAL=ctx.is_causal
                )

            with nvtx.range("backward.FA2Triton.kernel_dq"):
                grid = (L // ctx.Q_TILE_SIZE + (L % ctx.Q_TILE_SIZE != 0), B)
                flash_bwd_dq_kernel[grid](
                    Q, K, V, grad_output,
                    D_mat, logsumexp,
                    dQ,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    K.stride(0), K.stride(1), K.stride(2),
                    V.stride(0), V.stride(1), V.stride(2),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
                    D_mat.stride(0), D_mat.stride(1), D_mat.stride(2),
                    logsumexp.stride(0), logsumexp.stride(1),
                    dQ.stride(0), dQ.stride(1), dQ.stride(2),
                    L, L,
                    1 / (ctx.D ** 0.5),
                    D=ctx.D,
                    Q_TILE_SIZE=ctx.Q_TILE_SIZE,
                    K_TILE_SIZE=ctx.K_TILE_SIZE,
                    IS_CAUSAL=ctx.is_causal
                )

        return dQ, dK, dV, None


@torch.inference_mode()
def fa2_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Run the Triton FA2 forward kernel for inference prefill only.

    The wrapper intentionally does not create an autograd context or retain
    forward tensors. ``q``, ``k``, and ``v`` use the flattened-head layout
    ``[batch_heads, sequence_length, head_dim]``.
    """
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have the same [batch_heads, sequence, head_dim] shape")
    if not q.is_cuda:
        raise ValueError("fa2_prefill requires CUDA tensors")

    batch_heads, sequence_length, head_dim = q.shape
    output = torch.empty_like(q)
    query_tile_size = 64
    key_tile_size = 64
    grid = (triton.cdiv(sequence_length, query_tile_size), batch_heads)

    flash_fwd_kernel[grid](
        q, k, v,
        output, q,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        q.stride(0), q.stride(1),
        sequence_length, sequence_length,
        1 / (head_dim ** 0.5),
        D=head_dim,
        Q_TILE_SIZE=query_tile_size,
        K_TILE_SIZE=key_tile_size,
        IS_CAUSAL=is_causal,
        STORE_L=False,
    )
    return output


@torch.inference_mode()
def fa2_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Run single-token attention against a contiguous KV cache.

    ``q`` must have shape ``[batch_heads, 1, head_dim]`` and both caches must
    have shape ``[batch_heads, context_length, head_dim]``. The cache is
    assumed to contain only valid tokens, including the current token's K/V.
    """
    if q.ndim != 3 or q.shape[1] != 1:
        raise ValueError("q must have shape [batch_heads, 1, head_dim]")
    if key_cache.ndim != 3 or value_cache.shape != key_cache.shape:
        raise ValueError("key_cache and value_cache must have the same 3D shape")
    batch_heads, _, head_dim = q.shape
    if (
        key_cache.shape[0] != batch_heads
        or key_cache.shape[2] != head_dim
        or key_cache.shape[1] < 1
    ):
        raise ValueError("cache shape must be [batch_heads, context_length, head_dim]")
    if not q.is_cuda or key_cache.device != q.device or value_cache.device != q.device:
        raise ValueError("q and KV caches must be CUDA tensors on the same device")
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two")

    output = torch.empty_like(q)
    context_length = key_cache.shape[1]
    key_tile_size = 64
    flash_decode_kernel[(batch_heads,)](
        q, key_cache, value_cache, output,
        q.stride(0), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        output.stride(0), output.stride(2),
        context_length,
        1 / (head_dim ** 0.5),
        D=head_dim,
        K_TILE_SIZE=key_tile_size,
    )
    return output

def main():
    B, L, D = 2, 8, 16
    Q = torch.randn(B, L, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, L, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, L, D, device='cuda', dtype=torch.float32)
    is_causal = True

    output = FA2Triton.apply(Q, K, V, is_causal)
    print(output.shape)

if __name__ == "__main__":
    main()