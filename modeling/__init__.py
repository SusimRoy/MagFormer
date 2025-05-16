# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .model import nformer_model


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(cfg, num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model

def build_nformer_model(cfg, num_classes):
    model = nformer_model(cfg, num_classes)
    if hasattr(cfg.MODEL, 'BACKBONE_WEIGHTS_PATH') and cfg.MODEL.BACKBONE_WEIGHTS_PATH:
        import torch
        backbone_weights = torch.load(cfg.MODEL.BACKBONE_WEIGHTS_PATH)
        model.backbone.load_state_dict(backbone_weights)
        
        # Freeze backbone if specified
        if hasattr(cfg.MODEL, 'FREEZE_BACKBONE') and cfg.MODEL.FREEZE_BACKBONE:
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen for training")
    return model

def save_backbone_weights(model, save_path):
    """Save the backbone weights after training
    
    Args:
        model: The trained baseline model
        save_path: Path to save the weights
    """
    import torch
    if hasattr(model, 'backbone'):
        # If the model is already the nformer model
        torch.save(model.backbone.state_dict(), save_path)
    else:
        # If the model is the backbone itself
        torch.save(model.state_dict(), save_path)
