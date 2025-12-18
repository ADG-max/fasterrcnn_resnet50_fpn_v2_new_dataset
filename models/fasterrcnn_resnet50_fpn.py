import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.custom_roi_heads import WeightedRoIHeads

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model
    
    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    class_weights = torch.tensor(
        [1.0, 1.8, 2.2, 2.2],  # background + classes
        dtype=torch.float32
    )
    class_weights = class_weights.to(
        next(model.parameters()).device
    )

    model.roi_heads = WeightedRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,
    
        # === DEFINE MANUALLY (TorchVision >= 0.15) ===
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
    
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    
        class_weights=class_weights
    )
    return model

if __name__ == '__main__':
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
