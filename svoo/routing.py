"""Original SVOO block-selection policy, shared by CUDA and NPU."""
import torch


def weighted_softmax(scores, weights):
    """Softmax over scores after weighting each key cluster by its token count."""
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)


def identify_dynamic_map(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
    max_kc_ratio=0,
):
    """
    Args:
        min_kc_ratio: scalar or per-head ratio [H] for minimum key clusters kept.
        max_kc_ratio: optional scalar or per-head cap ratio. 0/None disables the cap.
                      Each query cluster keeps at most this ratio of key clusters.
    """
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    if p >= 1.0:
        return torch.ones(B, H, qc_num, kc_num, dtype=torch.bool, device=device)

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if isinstance(min_kc_ratio, (list, tuple)) or (hasattr(min_kc_ratio, '__len__') and len(min_kc_ratio) == H):
        if not isinstance(min_kc_ratio, torch.Tensor):
            min_kc_ratio = torch.tensor(min_kc_ratio, device=device, dtype=torch.float32)
        for h in range(H):
            head_ratio = min_kc_ratio[h].item()
            if head_ratio > 0:
                preserve_length = int(head_ratio * kc_num)
                remove_indices[:, h, :, :preserve_length] = False
    elif isinstance(min_kc_ratio, (int, float)) and min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    if max_kc_ratio is not None:
        if isinstance(max_kc_ratio, (list, tuple)) or (hasattr(max_kc_ratio, "__len__") and len(max_kc_ratio) == H):
            if not isinstance(max_kc_ratio, torch.Tensor):
                max_kc_ratio = torch.tensor(max_kc_ratio, device=device, dtype=torch.float32)
            for h in range(H):
                head_ratio = float(max_kc_ratio[h].item())
                if head_ratio > 0:
                    cap = int(head_ratio * kc_num)
                    cap = max(1, cap)
                    if isinstance(min_kc_ratio, torch.Tensor) and min_kc_ratio.numel() == H:
                        cap = max(cap, int(float(min_kc_ratio[h].item()) * kc_num))
                    remove_indices[:, h, :, cap:] = True
        elif isinstance(max_kc_ratio, (int, float)) and float(max_kc_ratio) > 0:
            cap = int(float(max_kc_ratio) * kc_num)
            cap = max(1, cap)
            if isinstance(min_kc_ratio, torch.Tensor) and min_kc_ratio.numel() == H:
                for h in range(H):
                    cap_h = cap
                    head_min = float(min_kc_ratio[h].item())
                    if head_min > 0:
                        cap_h = max(cap_h, int(head_min * kc_num))
                    cap_h = max(1, cap_h)
                    remove_indices[:, h, :, cap_h:] = True
            else:
                if isinstance(min_kc_ratio, (int, float)) and float(min_kc_ratio) > 0:
                    cap = max(cap, int(float(min_kc_ratio) * kc_num))
                remove_indices[..., cap:] = True

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map
