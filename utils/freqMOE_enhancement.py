"""基于频域划分的 freqMoE 增强 - 改进版。

改进点：
1. 使用Soft Frequency Mask（可微的sigmoid窗函数）替代hard mask
2. 支持Frequency-wise Routing（频点级MoE）
3. 支持低通滤波器预处理（先滤高频，再在低频上做MoE）
4. 负载均衡loss防止专家塌缩
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqMoEEnhancer(nn.Module):
    """将输入序列转换到频域并通过频段专家重加权。

    改进版使用Soft Frequency Mask实现可微的频段划分。
    支持低通滤波器预处理：先滤除高频，再在低频部分应用MoE专家划分。
    支持细粒度低频专家：将专家集中在低频范围内进行更细的划分。
    """

    def __init__(
        self,
        seq_len: int,
        num_experts: int = 3,
        use_norm: bool = True,
        use_soft_mask: bool = True,
        sharpness: float = 20.0,
        use_freq_wise_routing: bool = False,
        # Low-pass filter parameters
        use_lowpass_filter: bool = False,
        lowpass_cutoff: float = 0.3,  # Ratio of freq_len to keep
        # Fine-grained low-freq experts parameter
        expert_freq_range: float = 1.0,  # Expert frequency range ratio (1.0 = full range, 0.3 = only 30% low freq)
    ) -> None:
        super().__init__()
        self.seq_len = max(1, int(seq_len))
        self.freq_len = self.seq_len // 2 + 1
        self.num_experts = max(1, int(num_experts))
        self.use_norm = bool(use_norm)
        self.use_soft_mask = bool(use_soft_mask)
        self.sharpness = float(sharpness)
        self.use_freq_wise_routing = bool(use_freq_wise_routing)

        # Low-pass filter parameters
        self.use_lowpass_filter = bool(use_lowpass_filter)
        self.lowpass_cutoff = float(lowpass_cutoff)

        # Fine-grained low-freq experts parameter
        self.expert_freq_range = float(expert_freq_range)

        if self.use_soft_mask:
            self._init_soft_mask_params()
        else:
            self.theta = nn.Parameter(torch.rand(self.num_experts - 1))

        if self.use_freq_wise_routing:
            self.gate = nn.Linear(self.freq_len, self.num_experts)
        else:
            self.gate = nn.Linear(self.freq_len, self.num_experts)

    def _init_soft_mask_params(self):
        """初始化可微的频段参数（中心位置和宽度）。

        初始化时将专家的频段均匀分布在指定的频率范围内。
        如果expert_freq_range < 1.0，则将专家集中在低频范围。
        """
        # 根据expert_freq_range确定中心点范围
        # 如果expert_freq_range=1.0: 中心点在[0.15, 0.85]
        # 如果expert_freq_range=0.3: 中心点在[0.15*0.3, 0.85*0.3] = [0.045, 0.255]
        center_min = 0.15 * self.expert_freq_range
        center_max = 0.85 * self.expert_freq_range
        centers = torch.linspace(center_min, center_max, self.num_experts)

        # 初始化宽度，使每个专家覆盖 expert_freq_range/num_experts 的范围
        widths = torch.ones(self.num_experts) * (
            self.expert_freq_range * 0.9 / self.num_experts
        )
        self.centers = nn.Parameter(centers)
        self.widths = nn.Parameter(widths)

    def _create_soft_mask(self, freq_coords: torch.Tensor) -> torch.Tensor:
        """创建可微的频段掩码。

        使用sigmoid函数创建平滑的频段边界，实现可微的频率划分。

        Args:
            freq_coords: 归一化的频率坐标 [F]

        Returns:
            masks: [num_experts, F] 每个专家的频段掩码
        """
        centers = torch.sigmoid(self.centers)
        widths = F.softplus(self.widths) + 1e-3

        masks = []
        for i in range(self.num_experts):
            center = centers[i]
            width = widths[i]

            left_edge = center - width / 2
            right_edge = center + width / 2

            mask = torch.sigmoid(
                (freq_coords - left_edge) * self.sharpness
            ) - torch.sigmoid((freq_coords - right_edge) * self.sharpness)
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def _create_hard_mask(self, freq_len: int) -> torch.Tensor:
        """创建hard mask（用于对比或作为备选）。

        如果expert_freq_range < 1.0，则将专家频段限制在低频范围内。

        Returns:
            masks: [num_experts, freq_len]
        """
        cuts = torch.sigmoid(self.theta) * self.expert_freq_range
        cuts, _ = torch.sort(cuts)

        cuts = torch.cat(
            [
                torch.tensor([0.0], device=cuts.device),
                cuts,
                torch.tensor([self.expert_freq_range], device=cuts.device),
            ]
        )
        bounds = (cuts * freq_len).long()
        bounds[-1] = int(self.expert_freq_range * freq_len)

        masks = []
        for idx in range(self.num_experts):
            start = bounds[idx].item()
            end = bounds[idx + 1].item()

            mask = torch.zeros(freq_len)
            if start < end:
                mask[start:end] = 1.0
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def forward(self, x):
        """前向传播。

        支持两种模式：
        1. 普通模式：直接对全频谱应用MoE专家划分
        2. 低通滤波模式：先滤除高频，再在低频部分应用MoE

        Returns:
            output: 增强后的时域信号
            weights: 专家权重 [B, num_experts] 或 [B, F, num_experts]
            masks: 频段掩码（用于分析）
        """
        mean = torch.zeros_like(x[:, :1, :])
        std = torch.ones_like(mean)
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-6
            x = (x - mean) / std

        x_fft = torch.fft.rfft(x, dim=1)
        x_fft = x_fft.permute(0, 2, 1)  # [B, F, C]
        freq_len = self.freq_len

        # 如果启用低通滤波器，先滤除高频
        if self.use_lowpass_filter:
            # 创建低通滤波器掩码
            lowpass_end = int(freq_len * self.lowpass_cutoff)
            lowpass_mask = torch.zeros(freq_len, device=x.device)
            lowpass_mask[:lowpass_end] = 1.0

            # 分离低频和高频部分
            x_fft_low = x_fft * lowpass_mask
            x_fft_high = x_fft * (1.0 - lowpass_mask)

            # 对低频部分应用MoE
            x_fft_for_moe = x_fft_low
            x_fft_high_residual = x_fft_high  # 保存高频残差
        else:
            x_fft_for_moe = x_fft
            x_fft_high_residual = None  # 没有高频残差

        freq_coords = torch.linspace(0, 1, freq_len, device=x.device)

        if self.use_soft_mask:
            # 对于低通模式，调整掩码的频率范围
            if self.use_lowpass_filter:
                # 重新归一化频率坐标到低频范围
                lowpass_end = int(freq_len * self.lowpass_cutoff)
                if lowpass_end > 1:
                    freq_coords_low = torch.linspace(0, 1, lowpass_end, device=x.device)
                    masks = self._create_soft_mask(freq_coords_low)
                    # 扩展到完整长度
                    full_masks = torch.zeros(
                        self.num_experts, freq_len, device=x.device
                    )
                    full_masks[:, :lowpass_end] = masks
                    masks = full_masks
                else:
                    masks = self._create_soft_mask(freq_coords)
            else:
                masks = self._create_soft_mask(freq_coords)
        else:
            masks = self._create_hard_mask(freq_len)

        masks = masks.to(x.device)

        if self.use_freq_wise_routing:
            # For frequency-wise routing: each frequency bin gets its own gate decision
            # gating_input: [B, F] - use absolute magnitude for each frequency bin (sum across channels)
            gating_input = torch.abs(x_fft_for_moe).sum(dim=-1)  # [B, F]
            # gate: [B, F] -> Linear(freq_len, num_experts) -> [B, F, num_experts]
            gate_output = self.gate(gating_input)
            weights = torch.softmax(gate_output, dim=-1)
            # weights: [B, F, num_experts]
        else:
            # Global routing: one gate decision per sample
            gating_input = torch.abs(x_fft_for_moe).mean(dim=1)
            gate_output = self.gate(gating_input)
            weights = torch.softmax(gate_output, dim=-1)

        # 获取实际处理的频率长度（低通模式下可能小于完整freq_len）
        actual_freq_len = x_fft_for_moe.shape[1]

        expert_components = []
        for i in range(self.num_experts):
            # 获取该专家的掩码 [freq_len] or [actual_freq_len]
            mask_i = masks[i, :actual_freq_len]  # [actual_freq_len]
            # 扩展掩码维度以匹配 x_fft_for_moe 的形状 [B, actual_freq_len, C]
            mask_i = mask_i.unsqueeze(0).unsqueeze(-1)  # [1, actual_freq_len, 1]
            expert_fft = x_fft_for_moe * mask_i
            expert_components.append(expert_fft)

        expert_components = torch.stack(expert_components, dim=-1)

        if self.use_freq_wise_routing:
            combined_fft = (expert_components * weights.unsqueeze(-2)).sum(dim=-1)
        else:
            # weights: [B, num_experts] -> [B, 1, 1, num_experts] for broadcasting
            weights_broadcast = weights.unsqueeze(1).unsqueeze(2)
            combined_fft = (expert_components * weights_broadcast).sum(dim=-1)

        # 如果启用了低通滤波，加上高频残差
        if self.use_lowpass_filter and x_fft_high_residual is not None:
            combined_fft = combined_fft + x_fft_high_residual
        else:
            pass  # No residual to add

        combined_fft = combined_fft.permute(0, 2, 1)

        output = torch.fft.irfft(combined_fft, n=self.seq_len, dim=1)

        if self.use_norm:
            output = output * std + mean

        if self.use_freq_wise_routing:
            final_weights = weights.permute(0, 2, 1)
        else:
            final_weights = weights.unsqueeze(1).unsqueeze(2)

        return output, final_weights, masks

    @torch.no_grad()
    def describe_expert_bounds(self):
        """Return current expert frequency intervals and coverage ratios."""
        freq_len = self.freq_len

        if self.use_soft_mask:
            centers = torch.sigmoid(self.centers.detach())
            widths = F.softplus(self.widths.detach()) + 1e-3

            segments = []
            for idx in range(self.num_experts):
                center = centers[idx].item()
                width = widths[idx].item()
                start = max(0, int((center - width / 2) * freq_len))
                end = min(freq_len, int((center + width / 2) * freq_len))
                coverage = (end - start) / max(1, freq_len)
                segments.append((idx, start, end, coverage))
            return segments
        else:
            if self.num_experts <= 1 or self.theta.numel() == 0:
                return [(0, 0, freq_len, 1.0)]
            cuts = torch.sigmoid(self.theta.detach()) * self.expert_freq_range
            cuts, _ = torch.sort(cuts)
            device = cuts.device
            cuts = torch.cat(
                [
                    torch.tensor([0.0], device=device),
                    cuts,
                    torch.tensor([self.expert_freq_range], device=device),
                ]
            )
            bounds = (cuts * freq_len).long()
            bounds[-1] = int(self.expert_freq_range * freq_len)
            segments = []
            for idx in range(self.num_experts):
                start = bounds[idx].item()
                end = bounds[idx + 1].item()
                if end <= start:
                    end = min(freq_len, start + 1)
                coverage = (end - start) / max(1, freq_len)
                segments.append((idx, start, end, coverage))
            return segments

    def get_load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失。

        使用KL散度使专家权重接近均匀分布，防止专家塌缩。

        Args:
            weights: 专家权重 [B, num_experts] 或 [B, F, num_experts]

        Returns:
            loss: 负载均衡损失
        """
        # Handle different weight shapes
        if weights is None:
            return torch.tensor(0.0, device="cpu")

        original_shape = weights.shape

        if weights.dim() == 4:
            # [B, 1, 1, N] -> [B, N]
            weights = weights.squeeze(1).squeeze(1)
        elif weights.dim() == 3:
            # [B, F, N] (freq-wise) or [B, 1, N]
            if weights.shape[1] == 1:
                weights = weights.squeeze(1)
            else:
                # For freq-wise routing, average across frequency dimension first
                weights = weights.mean(dim=1)

        # Now weights should be [B, N]
        if weights.dim() != 2:
            return torch.tensor(0.0, device="cpu")

        avg_weights = weights.mean(dim=0)

        # Add small epsilon for numerical stability
        avg_weights = avg_weights + 1e-8
        avg_weights = avg_weights / avg_weights.sum()

        uniform = torch.ones_like(avg_weights) / self.num_experts

        # Use KL divergence
        loss = F.kl_div(torch.log(avg_weights + 1e-12), uniform, reduction="batchmean")

        return loss


class FreqMoEWithBalanceLoss(nn.Module):
    """FreqMoE增强器，带有可选择的负载均衡损失。"""

    def __init__(
        self,
        seq_len: int,
        num_experts: int = 3,
        use_norm: bool = True,
        use_soft_mask: bool = True,
        sharpness: float = 20.0,
        use_freq_wise_routing: bool = False,
        balance_loss_weight: float = 0.01,
        # Low-pass filter parameters
        use_lowpass_filter: bool = False,
        lowpass_cutoff: float = 0.3,
        # Fine-grained low-freq experts
        expert_freq_range: float = 1.0,
    ):
        super().__init__()
        self.freq_moe = FreqMoEEnhancer(
            seq_len=seq_len,
            num_experts=num_experts,
            use_norm=use_norm,
            use_soft_mask=use_soft_mask,
            sharpness=sharpness,
            use_freq_wise_routing=use_freq_wise_routing,
            use_lowpass_filter=use_lowpass_filter,
            lowpass_cutoff=lowpass_cutoff,
            expert_freq_range=expert_freq_range,
        )
        self.balance_loss_weight = balance_loss_weight

    def forward(self, x):
        return self.freq_moe(x)

    def get_total_loss(self, x, criterion):
        """计算总损失 = 重构损失 + 负载均衡损失。"""
        output, weights, masks = self.freq_moe(x)
        rec_loss = criterion(output, x)

        if self.balance_loss_weight > 0:
            bal_loss = self.freq_moe.get_load_balance_loss(weights)
            total_loss = rec_loss + self.balance_loss_weight * bal_loss
            return total_loss, rec_loss, bal_loss

        return rec_loss, rec_loss, None
