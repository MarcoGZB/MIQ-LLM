import torch
import torch.nn as nn
import torch.nn.functional as F

class FaCBLoss(nn.Module):
    """
    Fast-adaptive Convergence Balancer (FaCB)
    """
    def __init__(self, num_tasks=2, temperature=2.0, momentum=0.9, device='cuda'):
        super(FaCBLoss, self).__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature # 用于平滑收敛分数的温度超参数
        self.momentum = momentum       # 动量更新系数，保证权重变化平滑
        self.device = device

        # 记录初始阶段的损失 L_i(0)，用于作为收敛参考基准
        self.initial_losses = None
        
        # 记录上一轮的动态权重，初始化为均等权重 (e.g., [1.0, 1.0])
        self.current_weights = torch.ones(num_tasks, device=device)
        
        # 微调阶段专用：锁定权重的标志位
        self.weights_frozen_for_finetune = False

    def compute_base_losses(self, logits_cls, labels_cls, preds_reg, labels_reg):
        """
        计算基础的任务损失
        Task 0: FD (Cross-Entropy)
        Task 1:  (MSE or L1)
        """
        # 1. 故障诊断分类损失
        loss_ce = F.cross_entropy(logits_cls, labels_cls)
        
        # 2. RUL回归预测损失
        loss_mse = F.mse_loss(preds_reg.squeeze(), labels_reg.float())
        
        return torch.stack([loss_ce, loss_mse])

    def _post_training_forward(self, current_losses):
        """
        Post-training
        """
        # 如果是第一个Batch，记录初始Loss作为基准基数 L(0)
        if self.initial_losses is None:
            self.initial_losses = current_losses.detach().clone()
            return torch.sum(current_losses * self.current_weights)

        # 1. 计算收敛率 (Convergence Ratio)
        # Ratio = L_i(t) / L_i(0)
        # Ratio 越小，说明该任务收敛越快（Loss下降快）；Ratio 越大，说明收敛越慢。
        convergence_ratios = current_losses.detach() / (self.initial_losses + 1e-8)

        # 2. 计算动态权重分配得分
        # 我们希望赋予收敛慢的任务（Ratio大的任务）更高的权重。
        # 使用 Softmax 进行归一化，并引入 Temperature 进行平滑。
        # 乘以 num_tasks 保持总权重和为 num_tasks。
        scores = F.softmax(convergence_ratios / self.temperature, dim=0)
        target_weights = scores * self.num_tasks

        # 3. 动量更新权重 (防止权重剧烈震荡，保证损失曲线平稳下降)
        self.current_weights = self.momentum * self.current_weights + (1 - self.momentum) * target_weights
        
        # 为了稳定，将权重限制在合理范围内 (例如 0.1 到 5.0 之间)
        self.current_weights = torch.clamp(self.current_weights, min=0.1, max=5.0)

        # 4. 计算加权总损失
        total_loss = torch.sum(current_losses * self.current_weights)
        
        return total_loss

    def _fine_tuning_forward(self, current_losses):
        """
        Fine-tuning
        """
        if not self.weights_frozen_for_finetune:
            # 冻结从 Post-training 学到的最终平衡状态
            # 也可以在这里引入微小的可学习噪声参数（参考不确定性权重的思路），但保持主导权重恒定
            self.frozen_weights = self.current_weights.detach().clone()
            self.weights_frozen_for_finetune = True
            
        # 使用冻结的鲁棒权重进行聚合，防止小样本带来的损失震荡
        total_loss = torch.sum(current_losses * self.frozen_weights)
        
        return total_loss

    def forward(self, logits_cls, labels_cls, preds_reg, labels_reg, phase="post_train"):
        """
        :param logits_cls: 分类任务输出 (Fault Diagnosis)
        :param labels_cls: 分类真实标签
        :param preds_reg: 回归任务输出 (RUL Assessment)
        :param labels_reg: 回归真实标签
        :param phase: 当前阶段 ("post_train" 或 "fine_tune")
        """
        # 1. 获得当前的 [L_ce, L_mse]
        current_losses = self.compute_base_losses(logits_cls, labels_cls, preds_reg, labels_reg)

        # 2. 根据训练阶段进入不同的平衡逻辑
        if phase == "post_train":
            total_loss = self._post_training_forward(current_losses)
        elif phase == "fine_tune":
            total_loss = self._fine_tuning_forward(current_losses)
        else:
            raise ValueError("Phase must be either 'post_train' or 'fine_tune'")

        # 记录日志字典方便追踪
        loss_dict = {
            "Total_Loss": total_loss.item(),
            "Loss_CE": current_losses[0].item(),
            "Loss_MSE": current_losses[1].item(),
            "Weight_CE": self.current_weights[0].item(),
            "Weight_MSE": self.current_weights[1].item(),
        }

        return total_loss, loss_dict