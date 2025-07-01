import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from:
    Khosla et al. (2020), "Supervised Contrastive Learning"
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, temperature=None, device=None):
        """
        Compute loss
        Args:
            features: [bsz, n_views, dim]
            labels: [bsz]
        """
        if temperature is not None:
            self.temperature = temperature

        if len(features.shape) < 3:
            raise ValueError("Expected 3D input tensor")

        bsz = features.shape[0]
        n_views = features.shape[1]

        features = F.normalize(features, dim=2)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != bsz:
            raise ValueError("Mismatch between labels and features")

        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(n_views, n_views)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(bsz * n_views).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, bsz).mean()

        return loss


# Define global loss_map
loss_map = {
    'CrossEntropyLoss': nn.CrossEntropyLoss(),
    'SupConLoss': SupConLoss()
}
