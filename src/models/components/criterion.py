import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def generate_cams(features, weights):
    """
    使用 F.conv2d 将 FC 权重视为 1x1 卷积核来生成 CAMs。
    Args:
        features (torch.Tensor): 空间特征图 (B, C, H, W)
        weights (torch.Tensor): 分类器权重 (K, C)
    Returns:
        torch.Tensor: 类激活图 (B, K, H, W)
    """
    # weights: (K, C) -> (K, C, 1, 1)
    # F.conv2d 将其视为 K 个 (in_channels=C, out_channels=1) 的 1x1 卷积核
    return F.conv2d(features, weights.unsqueeze(-1).unsqueeze(-1))

def CAT_loss(cam_s, cam_t, pool_size=2):
    """
    计算 CAT-KD 损失 (池化, L2归一化, MSE)。
    """
    # 1. 池化
    if pool_size > 0:
        pool = F.adaptive_avg_pool2d
        cam_s = pool(cam_s, (pool_size, pool_size))
        cam_t = pool(cam_t, (pool_size, pool_size))
        
    # 2. L2 归一化 (在空间 H*W 维度上)
    cam_s_norm = F.normalize(cam_s.flatten(2), p=2, dim=2).view_as(cam_s)
    cam_t_norm = F.normalize(cam_t.flatten(2), p=2, dim=2).view_as(cam_t)
    
    # 3. MSE 损失
    loss = F.mse_loss(cam_s_norm, cam_t_norm)
    return loss

class KDCriterion:
    def __init__(self, **kwargs) -> None:
        args = SimpleNamespace(**kwargs)
        self.args = args
        # self.criterion_aligned_img_kd = args.img_criterion
        self.criterion_nlp_kd = args.nlp_criterion
        self.temperature = args.temperature #2
        self.cat_pool_size = args.cat_pool_size #2

        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = False)
        self.logit_scale = logit_scale.exp()

    def __call__(self, inputs):
        # 1. 解析所有张量
        (
            spatial_features_s,   # (B, C_student, H, W)
            student_cls_weights,  # (K_classes, C_student)
            hidden_features,      # (B, C_student) - 学生全局特征
            out,                  # (B, K_classes) - 学生 logits
            clip_img_features,    # (B, C_teacher)
            clip_nlp_features,    # (K_classes, C_teacher)
            aligned_img,          # (B, C_student) - (原 Lvis 目标)
            aligned_nlp           # (K_classes, C_student) - (我们的 CAT-KD "教师权重")
        ) = inputs
        #计算新的视觉蒸馏损失
        cam_s = generate_cams(spatial_features_s, student_cls_weights)
        cam_t_guided = generate_cams(spatial_features_s, aligned_nlp)
        cat_loss = CAT_loss(cam_s, cam_t_guided, pool_size=self.cat_pool_size)

        # img_loss = self.criterion_aligned_img_kd(hidden_features, aligned_img) 被cat_loss取代

        student_nlp_logits = self.logit_scale * hidden_features @ aligned_nlp.T / self.temperature
        teacher_nlp_logits = self.logit_scale * clip_img_features @ clip_nlp_features.T / self.temperature
        kd_loss = self.criterion_nlp_kd(F.log_softmax(student_nlp_logits, dim=1),
                             F.softmax(teacher_nlp_logits, dim=1)) * (self.temperature * self.temperature)
        kd_loss = kd_loss * self.args.class_num / 2
        
        # return img_loss, kd_loss
        return cat_loss, kd_loss