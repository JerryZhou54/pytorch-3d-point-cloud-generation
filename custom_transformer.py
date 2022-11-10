from transformers import ViTModel, ViTFeatureExtractor, ViTForMaskedImageModeling
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchsummary import summary

def pixel_bias(outViewN, outW, outH, renderDepth):
    initTile = torch.cat([
        #X.repeat([outViewN, 1, 1]), # [V,H,W]
        #Y.repeat([outViewN, 1, 1]), # [V,H,W]
        torch.ones([outViewN, outH, outW]).float() * renderDepth, 
        torch.zeros([outViewN, outH, outW]).float(),
    ], dim=0) # [4V,H,W]

    return initTile.unsqueeze_(dim=0) # [1,4V,H,W]

def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

class Custom_Transformer(nn.Module):
  def __init__(self, outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(Custom_Transformer, self).__init__()
        self.outViewN = outViewN
        self.outW = outW
        self.outH = outH
        self.renderDepth = renderDepth
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.transformer = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224")
        num_patches = (self.transformer.config.image_size // self.transformer.config.patch_size) ** 2
        self.bool_masked_pos = torch.zeros((1, num_patches)).bool().to("cuda:0")
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.transformer.vit.encoder.layer[11].requires_grad = True
        self.transformer.vit.embeddings.mask_token.requires_grad = True
        self.transformer.decoder.requires_grad = True
        self.conv1 = conv2d_block(3, 64)
        self.conv2 = conv2d_block(64, 128)
        self.fc1 = linear_block(25*25, 480)
        self.fc2 = linear_block(480, 384)
        self.fc3 = linear_block(384, 256)
        self.fc4 = linear_block(256, 128)
        self.fc5 = linear_block(128, 64)
        self.deconv1 = deconv2d_block(128, 96)
        self.deconv2 = deconv2d_block(96, 64)
        self.deconv3 = deconv2d_block(64, 32)
        self.deconv4 = deconv2d_block(32, 16)
        self.pixel_conv = nn.Conv2d(16, outViewN*2, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)
        self.relu = nn.ReLU()

  def forward(self, x):
    x = np.split(np.squeeze(np.array(x.cpu())), x.shape[0])
    for index, array in enumerate(x):
        x[index] = np.squeeze(array)
    x = torch.tensor(np.stack(self.feature_extractor(x)['pixel_values'], axis=0)).to("cuda:0")
    #x = self.feature_extractor(x, return_tensors="pt").pixel_values
    x = self.transformer(x, bool_masked_pos=self.bool_masked_pos).logits
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.fc1(x.view(-1, 25*25))
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    x = x.view(-1, 128, 8, 8)
    x = self.deconv1(F.interpolate(x, scale_factor=2))
    x = self.deconv2(F.interpolate(x, scale_factor=2))
    x = self.deconv3(F.interpolate(x, scale_factor=2))
    x = self.deconv4(F.interpolate(x, scale_factor=2))
    x = self.pixel_conv(x) + self.pixel_bias.to(x.device)
    depth, maskLogit = torch.split(
        x, [self.outViewN, self.outViewN], dim=1)

    return depth, maskLogit


#arr = torch.randn(1,3,64,64).to("cuda:0")
model = Custom_Transformer().to("cuda:0")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)
"""
with torch.no_grad():
  depth, mask = model(arr)
print(mask.shape)
print(depth.shape)
"""

#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Input shape: (3,224,224)
# Output shape: (24,128,128)
# Mask shape: (8,128,128)

#with torch.no_grad():
  #arr = feature_extractor(arr, return_tensors="pt")
  #output = model(arr)
#print(output.pooler_output.shape)
#print(output.last_hidden_state.shape)