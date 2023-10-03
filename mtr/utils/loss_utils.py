# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs


def ctrl_loss_gmm_direct(pred_scores, pred_ctrl, gt_ctrl, gt_valid_mask, log_std_range=(-4.0, 4.0), rho_limit=0.4):
    """
    GMM Loss for Control Action
    Args:
        pred_scores (batch_size, num_modes):
        pred_ctrl (batch_size, num_modes, 6 or 9)
        gt_ctrl (batch_size, 3 - dx, dy, dtheta):
        gt_valid_mask (batch_size):
        timestamp_loss_weight (num_timestamps):
    Returns:
        reg_loss (batch_size, num_modes):
    """
    independent = pred_ctrl.shape[-1] == 6
       
    gt_ctrl = gt_ctrl.unsqueeze(-2) # (batch_size, 1, 3)
    gt_valid_mask = gt_valid_mask.unsqueeze(-1) # (batch_size, 1)
    
    res_trajs = gt_ctrl - pred_ctrl[..., :3]  # (batch_size, num_modes, 3)
    res_trajs = res_trajs.unsqueeze(-1) # (batch_size, num_modes, 3, 1)
    
    # assert torch.isfinite(res_trajs).all(), "Nan in res_trajs"
    
    log_std1 = torch.clip(pred_ctrl[..., 3], min=log_std_range[0], max=log_std_range[1])
    log_std2 = torch.clip(pred_ctrl[..., 4], min=log_std_range[0], max=log_std_range[1])
    log_std3 = torch.clip(pred_ctrl[..., 5], min=log_std_range[0], max=log_std_range[1])
    std1 = torch.exp(log_std1)
    std2 = torch.exp(log_std2)
    std3 = torch.exp(log_std3) 

    if independent:
        rho1 = rho2 = rho3 = torch.zeros_like(log_std1)
    else:
        rho1 = torch.clip(pred_ctrl[..., 6], min=-rho_limit, max=rho_limit) # 1&2
        rho2 = torch.clip(pred_ctrl[..., 7], min=-rho_limit, max=rho_limit) # 1&3
        rho3 = torch.clip(pred_ctrl[..., 8], min=-rho_limit, max=rho_limit) # 2&3

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    
    R_det = 1 - rho1**2 - rho2**2 - rho3**2 + 2*rho1*rho2*rho3 # (batch_size, num_modes)
    # print('rdet', R_det.min(), R_det.max())
    # assert torch.isfinite(R_det).all(), "Nan in R_det"
    
    reg_gmm_log_coefficient = log_std1 + log_std2 + log_std3 \
        + 0.5 * torch.log(R_det)  # (batch_size, num_modes)
    
    # assert torch.isfinite(reg_gmm_log_coefficient).all(), "Nan in reg_gmm_log_coefficient"
    
    # Express the inverse of the covariance matrix in terms of the correlation coefficients
    cov_inv = torch.zeros((res_trajs.shape[0], res_trajs.shape[1], 3, 3)).type_as(res_trajs)
    
    cov_inv[..., 0, 0] = (1-rho3**2)/(std1**2)
    cov_inv[..., 0, 1] = (rho2*rho3-rho1)/(std1*std2)
    cov_inv[..., 0, 2] = (rho1*rho3-rho2)/(std1*std3)
    
    cov_inv[..., 1, 0] = (rho2*rho3-rho1)/(std1*std2)
    cov_inv[..., 1, 1] = (1-rho3**2)/(std2**2)
    cov_inv[..., 1, 2] = (rho1*rho2-rho3)/(std2*std3)
    
    cov_inv[..., 2, 0] = (rho1*rho3-rho2)/(std1*std3)
    cov_inv[..., 2, 1] = (rho1*rho2-rho3)/(std2*std3)
    cov_inv[..., 2, 2] = (1-rho2**2)/(std3**2)
    
    cov_inv = cov_inv/(R_det.unsqueeze(-1).unsqueeze(-1))
    # assert torch.isfinite(cov_inv).all(), "Nan in cov_inv"
    # print('cov_inv', cov_inv.min(), cov_inv.max())    
    reg_gmm_exp = 0.5 * torch.einsum(
            'bmij, bmjk->bmik',
            torch.einsum('bmij, bmik->bmjk', res_trajs, cov_inv),
            res_trajs
        ).squeeze(-1).squeeze(-1) # (batch_size, num_modes)
    
    # print('reg_gmm_exp', reg_gmm_exp.min(), reg_gmm_exp.max())    
    # assert torch.isfinite(reg_gmm_exp).all(), "Nan in reg_gmm_exp"
    
    reg_loss = (reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask
    
    # Choose the smallest NLL loss
    reg_loss, best_idx = torch.min(reg_loss, dim=-1)
    
    return reg_loss, best_idx