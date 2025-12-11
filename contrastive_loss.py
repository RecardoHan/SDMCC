import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_cosine_similarity_with_min_center(A, B):
    """
    归一化余弦相似度（最小值中心化）：
    S(i, j) = sum_k [(A_i(k) - min) * (B_j(k) - min)] / (sqrt(sum_k (A_i(k) - min)^2) * sqrt(sum_k (B_j(k) - min)^2))
    """
    X_min = torch.cat([A, B], dim=0).min(dim=0, keepdim=True)[0]  # [1, D]
    A_center = A - X_min  # [N, D]
    B_center = B - X_min  # [N, D]
    # 分子
    num = torch.matmul(A_center, B_center.t())  # [N, N]
    # 分母
    denom_A = torch.norm(A_center, dim=1, keepdim=True)  # [N, 1]
    denom_B = torch.norm(B_center, dim=1, keepdim=True)  # [N, 1]
    denom = torch.matmul(denom_A, denom_B.t()) + 1e-12  # [N, N]
    S = num / denom  # [N, N]
    return S
#实例级对比损失
class InstanceContrastiveLoss(nn.Module):
    """
    实例级对比损失，使用归一化余弦相似度（最小值中心化）。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        # features: [batch_size, 2, dim]
        batch_size = features.shape[0]
        assert features.shape[1] == 2, "Input features shape should be [batch, 2, ...]"
        device = features.device
        # features: [batch_size, 2, dim]
        p_a = features[:, 0]  # [N, D]
        p_b = features[:, 1]  # [N, D]

        # 使用自定义相似度
        sim_aa = normalized_cosine_similarity_with_min_center(p_a, p_a) / self.temperature
        sim_ab = normalized_cosine_similarity_with_min_center(p_a, p_b) / self.temperature
        sim_bb = normalized_cosine_similarity_with_min_center(p_b, p_b) / self.temperature
        sim_ba = normalized_cosine_similarity_with_min_center(p_b, p_a) / self.temperature

        #计算正对（正样本 i）与所有负对的归一化概率
        # l^a_i
        exp_sim_aa = torch.exp(sim_aa)
        exp_sim_ab = torch.exp(sim_ab)
        denom_a = exp_sim_aa.sum(dim=1) + exp_sim_ab.sum(dim=1)
        l_a = -torch.log(torch.exp(sim_ab.diag()) / denom_a + 1e-12)

        # l^b_i
        exp_sim_bb = torch.exp(sim_bb)
        exp_sim_ba = torch.exp(sim_ba)
        denom_b = exp_sim_bb.sum(dim=1) + exp_sim_ba.sum(dim=1)
        l_b = -torch.log(torch.exp(sim_ba.diag()) / denom_b + 1e-12)

        loss = (l_a + l_b).mean() / 2.0
        return loss


class ClusterContrastiveLossWithEntropy(nn.Module):
    """
    聚类级对比损失 + 熵正则项，使用归一化余弦相似度（最小值中心化）。
    输入：聚类概率（soft label），形状 [batch, 2, n_clusters]
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, cluster_probs):
        # cluster_probs: [batch_size, 2, K]
        batch_size, _, K = cluster_probs.shape
        device = cluster_probs.device

        q_a = cluster_probs[:, 0]  # [N, K]
        q_b = cluster_probs[:, 1]  # [N, K]

        # 使用自定义相似度
        sim_aa = normalized_cosine_similarity_with_min_center(q_a, q_a) / self.temperature
        sim_ab = normalized_cosine_similarity_with_min_center(q_a, q_b) / self.temperature
        sim_bb = normalized_cosine_similarity_with_min_center(q_b, q_b) / self.temperature
        sim_ba = normalized_cosine_similarity_with_min_center(q_b, q_a) / self.temperature

        # l^a_i
        exp_sim_aa = torch.exp(sim_aa)
        exp_sim_ab = torch.exp(sim_ab)
        denom_a = exp_sim_aa.sum(dim=1) + exp_sim_ab.sum(dim=1)
        l_a = -torch.log(torch.exp(sim_ab.diag()) / denom_a + 1e-12)

        # l^b_i
        exp_sim_bb = torch.exp(sim_bb)
        exp_sim_ba = torch.exp(sim_ba)
        denom_b = exp_sim_bb.sum(dim=1) + exp_sim_ba.sum(dim=1)
        l_b = -torch.log(torch.exp(sim_ba.diag()) / denom_b + 1e-12)

        # 熵正则项 H(D)
        D_a = cluster_probs[:, 0]
        D_b = cluster_probs[:, 1]
        P_q_a = D_a.sum(dim=0) / D_a.sum()
        P_q_b = D_b.sum(dim=0) / D_b.sum()
        H_D = -(P_q_a * torch.log(P_q_a + 1e-12)).sum() - (P_q_b * torch.log(P_q_b + 1e-12)).sum()

        loss = (l_a + l_b).mean() / 2.0 - H_D

        return loss
