import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        
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
        print("attention input: ", x.size())
        print("batch size: %d\nchannels: %d\nwidth: %d\nheight: %d" % (m_batchsize, C, width, height))

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        print("query size: ", proj_query.size())

        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        print("key size: ", proj_key.size())

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        print("energy size: ", energy.size())

        attention = self.softmax(energy) # BX (N) X (N) 
        print("attention size: ", attention.size())

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        print("value size: ", proj_value.size())
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        print("attention bmm out size: ", out.size())
       
        out = out.view(m_batchsize,C,width,height)
        print("attention reshape out size: ", out.size())

        out = self.gamma*out + x
        print("attention output size: ", out.size())

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
