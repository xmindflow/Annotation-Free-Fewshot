import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.dim = F_l
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        # self.conv1_1 = nn.Conv2d(2*F_l, F_l, kernel_size=1,stride=1,padding=0,bias=True)
        self.conv3d = nn.Conv3d(2, 1, 3, padding=1)

    def forward(self,g,x, pad = (0, 1, 0, 1)):
        x = torch.concat([x[:,:self.dim].unsqueeze(dim = 1), x[:,self.dim:].unsqueeze(dim = 1)], dim=1)
        x = self.conv3d(x).squeeze(dim = 1)        
        x = F.pad(x, pad, mode='replicate')
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class AdaptiveLKA(nn.Module):
    def __init__(self, dim, use3d = False):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.use3d = use3d
        if use3d:
            self.conv3d = nn.Conv3d(2, 1, 3, padding=1)

    def forward(self, x):
        if self.use3d:
            x = torch.concat([x[:,:self.dim].unsqueeze(dim = 1), x[:,self.dim:].unsqueeze(dim = 1)], dim=1)
            x = self.conv3d(x).squeeze(dim = 1)
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
    
class LKA_decoder(nn.Module):
    def __init__(self, channels=[2048,1024,512,256]):
        super(LKA_decoder,self).__init__()
        self.channels = channels
        self.Conv_1x1 = nn.Conv2d(2*channels[0],2*channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])

        self.ALKA4 = AdaptiveLKA(dim=channels[0], use3d = True)
        self.ALKA3 = AdaptiveLKA(dim=channels[1])
        self.ALKA2 = AdaptiveLKA(dim=channels[2])
        self.ALKA1 = AdaptiveLKA(dim=channels[3])

        
        self.SA = SpatialAttention()
        self.Upf = up_conv(ch_in=channels[3],ch_out=32)
        self.decoderf = nn.Sequential(nn.Conv2d(32, 16, (3, 3), padding=(1, 1), bias=True),
                                nn.ReLU(),
                                nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True))
    def forward(self,x, skips):
        d4 = self.Conv_1x1(x)
        # CAM4
        d4 = self.ALKA4(d4)

        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        # aggregate 3
        d3 = d3 + x3
        
            
        # CAM3
        d3 = self.ALKA3(d3)      
        d3 = self.ConvBlock3(d3)
                 
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1], pad = (0, 2, 0, 2))
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        d2 = self.ALKA2(d2)

        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2], pad = (0, 4, 0, 4))
        
        # aggregate 1
        d1 = d1 + x1

        # CAM1

        d1 = self.ALKA1(d1)
        d1 = self.ConvBlock1(d1)

        d1 = self.decoderf(self.Upf(d1))

        return d1
    