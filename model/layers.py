import  torch
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import math

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class MLP(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,1))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_fusion(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,320))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_fusion_gate(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_fusion_gate, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,768))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class cnn_extractor(torch.nn.Module):
    def __init__(self,input_size,feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs =torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size,feature_num,kernel)
            for kernel,feature_num in feature_kernel.items()]
        )
    def forward(self,input_data):
        input_data = input_data.permute(0, 2, 1)
        feature = [conv(input_data)for conv in self.convs]
        feature = [torch.max_pool1d(f,f.shape[-1])for f in feature]
        feature = torch.cat(feature,dim = 1)
        feature = feature.view([-1,feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    def __init__(self,input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim,1)

    def forward(self,input,mask):
        score = self.Line(input).view(-1,input.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score,input).squeeze(1)
        return output

class Resnet(torch.nn.Module):
    def __init__(self,out_channels):
        super(Resnet, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out