"""Model builders reconstructed from the uploaded notebooks.

Heavy dependencies are imported lazily. Full model construction needs
segmentation-models-pytorch and, for Prithvi, TerraTorch/PEFT.
"""
from __future__ import annotations
from typing import Sequence

DEFAULT_BANDS = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]
LORA_CONFIG = {"method":"LORA","replace_qkv":"qkv","peft_config_kwargs":{
    "target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],
    "lora_alpha":16,"r":16}}


def build_gid_resnet(num_classes: int = 16, encoder_weights=None):
    import segmentation_models_pytorch as smp
    return smp.Unet('resnet50', encoder_weights=encoder_weights, in_channels=3,
                    classes=num_classes, decoder_use_batchnorm=True)


def build_yrb_resnet(gid_checkpoint: str, num_classes: int = 7):
    """Build the six-band ResNet/U-Net arm exactly as in the LOBO notebooks."""
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    model=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    ckpt=torch.load(gid_checkpoint,map_location='cpu')
    state=ckpt['model_state'] if isinstance(ckpt,dict) and 'model_state' in ckpt else ckpt
    state=dict(state)
    weight=state['encoder.conv1.weight']
    state['encoder.conv1.weight']=torch.cat(
        [weight,weight.mean(1,keepdim=True).repeat(1,3,1,1)],dim=1)
    model.load_state_dict(state,strict=False)
    model.segmentation_head[0]=nn.Conv2d(model.segmentation_head[0].in_channels,
                                         num_classes,3,padding=1)
    return model


def build_prithvi(num_classes: int = 7, bands: Sequence[str] = DEFAULT_BANDS):
    from terratorch.models import EncoderDecoderFactory
    return EncoderDecoderFactory().build_model(
        task='segmentation', backbone='terratorch_prithvi_eo_v2_300',
        backbone_pretrained=True, backbone_bands=list(bands),
        necks=[{"name":"SelectIndices","indices":[-1]},
               {"name":"ReshapeTokensToImage"}],
        decoder='UperNetDecoder', decoder_channels=256,
        num_classes=num_classes, head_dropout=0.1, peft_config=LORA_CONFIG)


def unwrap_output(output):
    return output.output if hasattr(output,'output') else output
