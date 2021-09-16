import torch
import torch.nn as nn
import numpy as np
import cv2
from util import util

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttention,self).__init__()
        self.channel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)

        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)

        energy =  torch.bmm(proj_query,proj_key) # transpose check

        attention = self.softmax(energy) # BX (N) X (N) 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
       
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x

        return out

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = nn.functional.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = nn.functional.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class SelfAttentionClassifier(nn.Module):
    def __init__(self, in_features, label_num, projector_dim=None, backbone=None):
        super().__init__()
        self.backbone = backbone
        self.projector_dim = projector_dim

        if projector_dim is not None:
            self.projector_layer = ProjectorBlock(in_features=in_features, out_features=projector_dim)
            self.attention_layer = SelfAttention(in_dim=projector_dim)
            self.fc_layer = torch.nn.Linear(projector_dim, label_num)
        else:
            self.attention_layer = SelfAttention(in_dim=in_features)
            self.fc_layer = torch.nn.Linear(in_features, label_num)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = torch.nn.Flatten()

    def forward(self, x):
        
        # Use classification backbone if available.
        if self.backbone is not None:
            x = self.backbone(x)

        # If available, project to an intermediate feature map.
        if self.projector_dim is not None:
            x = self.projector_layer(x)

        # Add attention layer.
        attention_out = self.attention_layer(x)
        # Add average pooling layer.
        x = self.avg_pool(attention_out)
        # Flatten and add final FC layer for classification.
        x = self.flatten_layer(x)
        x = self.fc_layer(x)

        return x, attention_out

class LinearAttentionClassifier(nn.Module):
    def __init__(self, in_features, label_num, projector_dim=512, backbone=None):
        super().__init__()
        self.backbone = backbone
        self.projector_dim = projector_dim

        self.dense = torch.nn.Conv2d(in_channels=in_features, out_channels=projector_dim, kernel_size=8, padding=0, bias=True)
        self.projector_layer = ProjectorBlock(in_features=in_features, out_features=projector_dim)
        self.attention_layer = LinearAttentionBlock(in_features=projector_dim)
        
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = torch.nn.Flatten()

        self.classify = torch.nn.Linear(in_features + projector_dim, label_num, bias=True)

    def forward(self, x):
        
        # Use classification backbone if available.
        if self.backbone is not None:
            l = self.backbone(x)

        # Add average pooling layer and flatten.
        x = self.avg_pool(l)
        x = self.flatten_layer(x)

        # Calculate input weights
        g = self.dense(l)

        # Project to an intermediate feature map.
        l = self.projector_layer(l)

        # Add attention layer.
        c, g = self.attention_layer(l, g)

        # Concatenate attention and backbone outputs
        g = torch.cat((g,x), dim=1)

        # Final FC layer for classification.
        x = self.classify(g)

        return x, c

def calculate_heatmap(data, attention):
    
    # post-process attention weights
    weights = torch.abs(attention)
    b, c, h, w = weights.size()
    weights = torch.sum(weights, dim=1).view(b, 1, h, w)

    # Convert tensors to numpy images
    img = util.tensor2im(data)
    weights = util.tensor2im(weights)

    # resize and convert to image 
    weights = cv2.resize(weights, (img.shape[0], img.shape[1]))
    #weights /= weights.max()
    #weights *= 255
    #weights =  255 - weights.astype('uint8')

    # generate heat maps 
    weights = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(weights, 0.3, img, 0.7, 0)
    return heatmap, weights