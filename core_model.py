" Wall Pathology Detection - By Ailton Dos Santos"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math



# Custom ArcFace Layer
class ArcFaceLayer(nn.Module):
    """
    Implementation of Large Margin Arc Distance.
    This replaces the standard linear output layer of a Neural Network.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scalar value (radius of the hypersphere)
        self.m = m  # Angular margin penalty

        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute cosine/sine values for the margin math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # 2. Calculate sine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # 3. Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Handle numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. One-hot encoding for the correct class margin
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 6. Apply margin scaling
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# Civil Engineering Model
class CivilPathologyModel(nn.Module):
    """
    The complete architecture combining a CNN Backbone with the ArcFace Head.
    """

    def __init__(self, num_classes=2):
        super(CivilPathologyModel, self).__init__()

        # Backbone: ResNet50
        resnet = models.resnet50(pretrained=True)

        # Remove original classifier
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.embedding_dim = 2048

        # ArcFace Head
        self.arcface_head = ArcFaceLayer(
            in_features=self.embedding_dim,
            out_features=num_classes,
            s=30.0,
            m=0.5
        )

    def forward(self, images, labels=None):
        # Extract features
        features = self.backbone(images)
        embedding = features.view(features.size(0), -1)

        # If training (labels provided), return ArcFace logits
        if labels is not None:
            return self.arcface_head(embedding, labels)

        # If inference (no labels), return just the normalized vector
        else:
            return F.normalize(embedding)