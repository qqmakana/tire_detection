# hubconf.py
import torch
import torchvision
def pretrainedVit(model_name='vit_b_16', pretrained=True, **kwargs):
  # Get the pretrained weights for the model
  pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  # Setup a ViT model instance with pretrained weights
  pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights, **kwargs)
  # Freeze the base parameters
  for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
  # Change the classifier head
  class_names = ['Bad_tire','Good_tire']
  pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names))
  return pretrained_vit

