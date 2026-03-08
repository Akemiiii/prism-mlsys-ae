# implements efficient moes
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCMoe(nn.Module):
    # Dispatch-Combine MoE with shared experts (每个token均经过所有共享专家)
    def __init__(
        self,
        n_experts: int,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        calculate_l_aux: bool,
        expert_loop_threshold: int = 128,  # 当flatten后B大于此阈值时，采用expert循环实现
        share_expert: int = 0,  # 共享专家的数量
    ):
        """
        Args:
            n_experts (int): 普通专家 (normal expert) 的数量
            input_size (int): 输入与输出的特征维度
            hidden_size (int): 将输入特征压缩到的隐藏维度
            intermediate_size (int): MLP中间层的维度
            topk (int): 每个 token 选择的普通专家个数（通常 k=1 或2）
            calculate_l_aux (bool): 是否计算辅助损失（仅针对普通专家部分）
            expert_loop_threshold (int): 当flatten后B大于该阈值时，
                                         采用循环的方式计算普通专家输出，否则使用向量化实现
            share_expert (int): 共享专家的数量。共享专家作用于所有 token，无需 gate 选择。
        """
        super().__init__()
        self.n_experts = n_experts
        self.share_expert = share_expert
        self.input_dim = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.topk = topk
        self.expert_loop_threshold = expert_loop_threshold

        # 如果输入与隐藏维度不一致，需要先投影到 hidden space
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.output_proj = nn.Linear(hidden_size, input_size)
        else:
            self.input_proj = None
            self.output_proj = None

        # Gate 层只用于普通专家的选择，其输出维度为 n_experts
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.alpha = 0.001
        self.beta = nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.uniform = torch.ones(n_experts) / n_experts  # [n_experts,]
        self.gate_bias_update = True

        # 普通专家的参数：
        #   gate_proj_weight:  (n_experts, hidden_size, intermediate_size)
        #   up_proj_weight:    (n_experts, hidden_size, intermediate_size)
        #   down_proj_weight:  (n_experts, intermediate_size, hidden_size)
        self.gate_proj_weight = nn.Parameter(
            torch.empty(n_experts, hidden_size, intermediate_size)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(n_experts, hidden_size, intermediate_size)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(n_experts, intermediate_size, hidden_size)
        )

        for weight in [self.gate_proj_weight, self.up_proj_weight]:
            nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(
            self.down_proj_weight, mode="fan_in", nonlinearity="relu"
        )

        # 初始化共享专家的参数（形状与普通专家一致）
        if share_expert > 0:
            self.shared_gate_proj_weight = nn.Parameter(
                torch.empty(share_expert, hidden_size, intermediate_size)
            )
            self.shared_up_proj_weight = nn.Parameter(
                torch.empty(share_expert, hidden_size, intermediate_size)
            )
            self.shared_down_proj_weight = nn.Parameter(
                torch.empty(share_expert, intermediate_size, hidden_size)
            )
            for weight in [self.shared_gate_proj_weight, self.shared_up_proj_weight]:
                nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(
                self.shared_down_proj_weight, mode="fan_in", nonlinearity="relu"
            )
        else:
            self.shared_gate_proj_weight = None
            self.shared_up_proj_weight = None
            self.shared_down_proj_weight = None

        # 辅助损失仅对普通专家部分计算
        self.calculate_l_aux = calculate_l_aux
        self.l_aux = None

    def get_aux_loss(self):
        """
        返回当前 forward pass 计算的辅助损失（包含 balance_loss 和 router_loss），仅针对普通专家。
        如果 forward 尚未调用，则返回 None。
        """
        return self.l_aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, d) 或 (B, seq_len, d)

        Returns:
            torch.Tensor: 输出张量，形状与输入一致
        """
        need_reshape = False
        if x.dim() == 3:
            B_orig, seq_len, _ = x.size()
            x = x.view(-1, self.input_dim)
            need_reshape = True
        B = x.size(0)

        # 将输入投影到隐藏空间
        if self.input_proj is not None:
            x_proj = self.input_proj(x)  # (B, hidden_size)
        else:
            x_proj = x

        # 计算路由 logits 和 softmax 后的概率（均在 hidden 域中计算）
        gate_logits = self.gate(x_proj)  # (B, n_experts)
        # gate_probs = F.softmax(gate_logits, dim=1)  # (B, n_experts)
        # 对每个 token 选择 top-k expert 及其概率
        # topk_probs: (B, topk), topk_indices: (B, topk)
        _, topk_indices = (gate_logits + self.beta.to(gate_logits.device)).topk(
            self.topk, dim=-1
        )
        topk_probs = gate_logits.gather(dim=-1, index=topk_indices)
        topk_probs = F.sigmoid(topk_probs).to(x.dtype)  # [B, topk]

        # 辅助损失计算（仅针对普通专家）
        if self.calculate_l_aux:
            eps = 1e-9
            # 将 topk_indices 转换为 one-hot 表示，形状 (B, topk, n_experts)
            one_hot = F.one_hot(topk_indices, num_classes=self.n_experts)
            # 计算每个 expert 被选择的次数
            load = one_hot.sum(dim=(0, 1)).float()  # 形状 (n_experts,)

            # 归一化后的负载分布（理论上理想值为 1/n_experts）
            m = load / (B * self.topk)

            load_cv = load.std() / (load.mean() + eps)
            imbalance_ratio = (load.max() - load.min()) / (load.mean() + eps)

            self.l_aux = {
                "load_cv": load_cv,
                "imbalance_ratio": imbalance_ratio,
            }

            if self.training and self.gate_bias_update:
                with torch.no_grad():
                    self.beta.copy_(
                        self.beta
                        - self.alpha * torch.sign(m - self.uniform.to(self.beta.device))
                    )

        # 1. 计算普通专家的输出（Gate选择的部分）
        if B > self.expert_loop_threshold or self.training:
            # 循环实现
            combined_normal = torch.zeros(
                B, self.hidden_size, device=x.device, dtype=x.dtype
            )
            for expert in range(self.n_experts):
                # 找出 token 的 topk 中是否选择了该普通专家
                mask = topk_indices == expert  # (B, topk)
                if not mask.any():
                    continue
                indices = mask.nonzero(
                    as_tuple=False
                )  # (N, 2): 第1列 token 下标，第2列对应的 topk 位置
                token_indices = indices[:, 0]
                slot_indices = indices[:, 1]

                x_selected = x_proj[token_indices]  # (N, hidden_size)
                probs_selected = topk_probs[token_indices, slot_indices]  # (N,)

                # 获取当前普通专家的参数
                gate_proj = self.gate_proj_weight[
                    expert
                ]  # (hidden_size, intermediate_size)
                up_proj = self.up_proj_weight[
                    expert
                ]  # (hidden_size, intermediate_size)
                down_proj = self.down_proj_weight[
                    expert
                ]  # (intermediate_size, hidden_size)

                gate_out = x_selected.matmul(gate_proj)  # (N, intermediate_size)
                up_out = x_selected.matmul(up_proj)  # (N, intermediate_size)
                hidden = F.silu(gate_out) * up_out  # (N, intermediate_size)
                expert_out = hidden.matmul(down_proj)  # (N, hidden_size)

                weighted_expert_out = expert_out * probs_selected.unsqueeze(1)
                combined_normal.index_add_(0, token_indices, weighted_expert_out)
        else:
            # 向量化实现普通专家输出
            selected_gate_proj_weight = self.gate_proj_weight[
                topk_indices
            ]  # (B, topk, hidden_size, intermediate_size)
            selected_up_proj_weight = self.up_proj_weight[
                topk_indices
            ]  # (B, topk, hidden_size, intermediate_size)
            selected_down_proj_weight = self.down_proj_weight[
                topk_indices
            ]  # (B, topk, intermediate_size, hidden_size)

            x_expanded = x_proj.unsqueeze(1).expand(B, self.topk, self.hidden_size)
            x_flat = x_expanded.reshape(B * self.topk, self.hidden_size)

            gate_proj_flat = selected_gate_proj_weight.reshape(
                B * self.topk, self.hidden_size, self.intermediate_size
            )
            up_proj_flat = selected_up_proj_weight.reshape(
                B * self.topk, self.hidden_size, self.intermediate_size
            )

            gate_out = torch.bmm(x_flat.unsqueeze(1), gate_proj_flat).squeeze(
                1
            )  # (B * topk, intermediate_size)
            up_out = torch.bmm(x_flat.unsqueeze(1), up_proj_flat).squeeze(
                1
            )  # (B * topk, intermediate_size)

            hidden = F.silu(gate_out) * up_out
            down_proj_flat = selected_down_proj_weight.reshape(
                B * self.topk, self.intermediate_size, self.hidden_size
            )
            expert_out_flat = torch.bmm(hidden.unsqueeze(1), down_proj_flat).squeeze(1)
            expert_out = expert_out_flat.view(B, self.topk, self.hidden_size)

            combined_normal = (expert_out * topk_probs.unsqueeze(2)).sum(dim=1)

        # 2. 计算共享专家的输出（所有 token 均经过所有共享专家）
        if self.share_expert > 0:
            # 使用 einsum 实现向量化计算
            # 对于每个 token (B, hidden_size) 分别计算每个共享专家：
            # shared_gate_out: (B, share_expert, intermediate_size)
            shared_gate_out = torch.einsum(
                "bh,ehm->bem", x_proj, self.shared_gate_proj_weight
            )
            shared_up_out = torch.einsum(
                "bh,ehm->bem", x_proj, self.shared_up_proj_weight
            )
            shared_hidden = (
                F.silu(shared_gate_out) * shared_up_out
            )  # (B, share_expert, intermediate_size)
            shared_expert_out = torch.einsum(
                "bem,emi->bei", shared_hidden, self.shared_down_proj_weight
            )  # (B, share_expert, hidden_size)
            # 这里采用求和聚合所有共享专家的输出，你也可以选择均值
            combined_shared = shared_expert_out.sum(dim=1)  # (B, hidden_size)
        else:
            combined_shared = 0

        # 合并普通专家的输出和共享专家的输出
        combined = combined_normal + combined_shared

        if self.output_proj is not None:
            output = self.output_proj(combined)
        else:
            output = combined

        if need_reshape:
            output = output.view(B_orig, seq_len, self.input_dim)

        return output