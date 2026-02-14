"""
Modified VGG16 for 32x32 CIFAR-10 native dimensions.
"""

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, layer_spec, num_classes=10, init_weights=False):
        super(VGG, self).__init__()

        layers = []
        in_channels = 3
        for l in layer_spec:
            if l == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU(),
                ]
                in_channels = l

        self.features = nn.Sequential(*layers)
        
        # Safely collapse the remaining 4x4 feature map to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Scaled down the classifier for CIFAR-10 to save memory and compute
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def vgg16(num_classes=10, init_weights=False):
    #The last two pooling layers are removed so that the spatial dimensions stop 
    #at 4x4 instead of 1x1, avoiding the TensorRT crash
    vgg16_cfg = [
    64, 64, "pool", # 32x32 -> 16x16
    128, 128, "pool", # 16x16 -> 8x8
    256, 256, 256, "pool", # 8x8 -> 4x4
    512, 512, 512,   # removed pool
    512, 512, 512,   # removed pool
    ]
    return VGG(vgg16_cfg, num_classes, init_weights)
