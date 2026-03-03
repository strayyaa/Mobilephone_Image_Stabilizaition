"""基于频域划分的 freqMoE 增强。"""

from __future__ import annotations

import torch
import torch.nn as nn


class FreqMoEEnhancer(nn.Module):
	"""将输入序列转换到频域并通过频段专家重加权。"""

	def __init__(self, seq_len: int, num_experts: int = 3, use_norm: bool = True) -> None:
		super().__init__()
		self.seq_len = max(1, int(seq_len))
		self.freq_len = self.seq_len // 2 + 1
		self.num_experts = max(1, int(num_experts))
		self.use_norm = bool(use_norm)
		
		
		self.theta = nn.Parameter(torch.rand(self.num_experts - 1))
		self.gate = nn.Linear(self.freq_len,self.num_experts)
	
		# self.gate = nn.Sequential(
        #     nn.Linear(self.num_experts, self.num_experts),
        #     nn.ReLU(),
        #     nn.Linear(self.num_experts, self.num_experts),
        # )

	def forward(self, x):
		# print(f"FreqMoEEnhancer 输入形状: {x.shape}")
		mean = torch.zeros_like(x[:, :1, :])
		std = torch.ones_like(mean)
		if self.use_norm:
			mean = x.mean(dim=1, keepdim=True)
			std = x.std(dim=1, keepdim=True) + 1e-6
			x = (x - mean) / std
		x  = torch.fft.rfft(x, dim=1)
		x = x.permute(0, 2, 1)  # [B, F, C]
		freq_len = self.freq_len
		
		cuts = torch.sigmoid(self.theta)
		cuts, _ = torch.sort(cuts)
		
		cuts = torch.cat([
            torch.tensor([0.0], device=cuts.device),
            cuts,
            torch.tensor([1.0], device=cuts.device)
        ])
		bounds = (cuts * freq_len).long()
		bounds[-1] = freq_len
		
		mask = []
		for idx in range(self.num_experts):
			start = bounds[idx].item()
			end = bounds[idx + 1].item() 
			
			freq_mask = torch.zeros_like(x)
			if start < end:
				freq_mask[:, :, start:end] = 1.0
			expert_component = x * freq_mask
			mask.append(expert_component.unsqueeze(-1)) # [B, C, F, 1]

		gating_input = torch.abs(x).mean(dim=1) # [B, F]

		weights = torch.softmax(self.gate(gating_input), dim=-1)  # [B, N]
		mask = torch.cat(mask,dim=-1)  # [B, C, F, N]
		weights = weights.unsqueeze(1).unsqueeze(2) # [B, 1, 1, N]
		
		combined_freq_output = (mask * weights).sum(dim=-1)  # [B, C, F]
		combined_freq_output = combined_freq_output.permute(0, 2, 1)  # [B, F, C]
		
		output = torch.fft.irfft(combined_freq_output, n=self.seq_len, dim=1)
		if self.use_norm:
			output = output * std + mean
		return output, weights, bounds


	## 输出初始的专家边界情况
	@torch.no_grad()
	def describe_expert_bounds(self):
		"""Return current expert frequency intervals and coverage ratios."""
		freq_len = self.freq_len
		if self.num_experts <= 1 or self.theta.numel() == 0:
			return [(0, 0, freq_len, 1.0)]
		cuts = torch.sigmoid(self.theta.detach())
		cuts, _ = torch.sort(cuts)
		device = cuts.device
		cuts = torch.cat([
			torch.tensor([0.0], device=device),
			cuts,
			torch.tensor([1.0], device=device),
		])
		bounds = (cuts * freq_len).long()
		bounds[-1] = freq_len
		segments = []
		for idx in range(self.num_experts):
			start = bounds[idx].item()
			end = bounds[idx + 1].item()
			if end <= start:
				end = min(freq_len, start + 1)
			coverage = (end - start) / max(1, freq_len)
			segments.append((idx, start, end, coverage))
		return segments
