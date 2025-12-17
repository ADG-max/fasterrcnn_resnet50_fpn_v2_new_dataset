import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads

class WeightedRoIHeads(RoIHeads):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(
            class_logits,
            labels,
            weight=self.class_weights
        )

        # box loss tetap sama
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        box_regression = box_regression.reshape(class_logits.size(0), -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1.0 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss
