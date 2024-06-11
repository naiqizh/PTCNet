# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


import copy
import logging


logger = logging.getLogger(__name__)

class MCCSA(nn.Module):
    def __init__(self, in_channels, num_heads, window_size, kernel_size=3):
        super(MCCSA, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        self.conv1x1_q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_v = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.depthwise_conv_q = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)
        self.depthwise_conv_k = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)
        self.depthwise_conv_v = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        D = H = W = int(round(N ** (1/3)))

        # 计算新的形状
        x = x.view(B_, D, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # 转换为 (B_, C, D, H, W)

        # 1x1x1卷积
        q = self.conv1x1_q(x)
        k = self.conv1x1_k(x)
        v = self.conv1x1_v(x)
        
        # 3x3x3深度卷积
        q = self.depthwise_conv_q(q)
        k = self.depthwise_conv_k(k)
        v = self.depthwise_conv_v(v)

        # 重塑形状
        q = rearrange(q, 'b c d h w -> b (d h w) c')
        k = rearrange(k, 'b c d h w -> b (d h w) c')
        v = rearrange(v, 'b c d h w -> b (d h w) c')

        # 计算注意力权重
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / (C ** 0.5)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, -1, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, N, N)

        attn = self.softmax(attn)

        # 输出
        out = torch.bmm(attn, v)
        out = rearrange(out, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        out = out.reshape(B_, N, C)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # 0 or window_size // 2
        self.mlp_ratio = mlp_ratio
        if self.shift_size != 0:
            assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # self.attn = WindowAttention(
        #     dim, window_size=self.window_size, num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        #self.attn = MCCSA(dim, dim)  # 使用MCCSA替代MSA
        self.attn = MCCSA(dim, num_heads, window_size)  # 使用MCCSA替代MSA

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if max(self.shift_size) > 0:
            S, H, W = self.input_resolution
            img_mask = torch.zeros((1, S, H, W, 1))  # 1 S H W 1
            s_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        s, h, w = self.input_resolution
        B, C, S, H, W = x.shape

        assert S == s and H == h and W == w, "input feature has wrong size"

        # 展平操作：将输入形状从 (B, C, S, H, W) 转换为 (B, S*H*W, C)
        x = rearrange(x, 'b c s h w -> b (s h w) c')

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # 周期性移位（cyclic shift）
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # 窗口划分（window partition）
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # 多头自注意力机制（W-MSA/SW-MSA）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口（merge windows）
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, S, H, W)

        # 逆向移位（reverse cyclic shift）
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # 重新变换为展平形状
        x = x.view(B, S * H * W, C)

        # 残差连接和 DropPath
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 重新排列为原始形状
        x = rearrange(x, 'b (s h w) c -> b c s h w', s=S, h=H, w=W)
        return x






class PatchPartition(nn.Module):
    def __init__(self, patch_size=(4, 4, 4)):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W, D = x.shape
        patches = x.unfold(2, self.patch_size[0], self.patch_size[0]) \
                    .unfold(3, self.patch_size[1], self.patch_size[1]) \
                    .unfold(4, self.patch_size[2], self.patch_size[2])
        patches = patches.contiguous().view(B, C, -1, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        patches = patches.permute(0, 2, 1, 3, 4, 5).contiguous()  # (B, num_patches, C, patch_size, patch_size, patch_size)
        patches = patches.view(-1, C, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        return patches

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(4, 4, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Conv3d后形状为(B, embed_dim, D//patch_size, H//patch_size, W//patch_size)
        x = self.proj(x)
        _, _, D_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, embed_dim, D_new*H_new*W_new) -> (B, D_new*H_new*W_new, embed_dim)
        x = self.norm(x)
        x = x.view(B, D_new, H_new, W_new, -1).permute(0, 4, 1, 2, 3)  # (B, D_new*H_new*W_new, embed_dim) -> (B, embed_dim, D_new, H_new, W_new)
        return x


    
class ReversibleFunction(nn.Module):
    def __init__(self, dim):
        super(ReversibleFunction, self).__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class RRM(nn.Module):
    def __init__(self, dim):
        super(RRM, self).__init__()
        self.rf1 = ReversibleFunction(dim // 2)
        self.rf2 = ReversibleFunction(dim // 2)

    def forward(self, x):
        c = x.shape[1] // 2
        x1, x2 = torch.split(x, c, dim=1)
        y1 = x1 + self.rf2(x2)
        y2 = x2 + self.rf1(y1)
        return torch.cat([y1, y2], dim=1)



class PTCBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(4, 4, 4), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,use_shift=False):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()  # 添加激活函数
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        self.rrm = RRM(dim)
        shift_size = tuple([w // 2 for w in window_size]) if use_shift else (0, 0, 0)
        self.attn = SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)

    def forward(self, x):
        # 分支1：卷积操作
        x1 = self.conv(x)
        x1=self.activation(x1)
        x1 = self.rrm(x1)
        
        # 分支2：Swin Transformer Block
        x2 = self.attn(x)
        
        # 合并两个分支的结果
        x = x1 + x2
        return x







class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, norm_layer=nn.LayerNorm, downsample=True):
        super().__init__()
        self.ptc_block1 = PTCBlock(dim, input_resolution, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, norm_layer=norm_layer, use_shift=False)
        self.ptc_block2 = PTCBlock(dim, input_resolution, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, norm_layer=norm_layer, use_shift=True)
        self.downsample = nn.Conv3d(dim, dim*2, kernel_size=3, stride=2, padding=1) if downsample else None

    def forward(self, x):
        x = self.ptc_block1(x)
        x = self.ptc_block2(x)
        skip_connection = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip_connection, x


class Bottleneck(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ptc_block = PTCBlock(dim, input_resolution, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, norm_layer=norm_layer, use_shift=False)

    def forward(self, x):
        return self.ptc_block(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, norm_layer=nn.LayerNorm, upsample=True):
        super().__init__()
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=1)
        self.ptc_block1 = PTCBlock(out_dim, input_resolution, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, norm_layer=norm_layer, use_shift=True)
        self.ptc_block2 = PTCBlock(out_dim, input_resolution, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, norm_layer=norm_layer, use_shift=False)

    def forward(self, x, skip_connection):
        x = x.contiguous()
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
            x = self.conv(x)
        x = (x + skip_connection).contiguous()
        x = self.ptc_block1(x).contiguous()
        x = self.ptc_block2(x).contiguous()
        return x
    
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, dim * (dim_scale ** 3)*2, bias=False)  # 扩展到 dim * 64*2
        self.norm = norm_layer(dim * 2)  # 扩展到 dim * 2

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)
        #print(f"x.flatten and permute: {x.shape}")  # torch.Size([B, D*H*W, C])
        x = self.expand(x)
        #print(f"After expand: {x.shape}")  # torch.Size([B, D*H*W, C*8])
        x = x.view(B, D, H, W, C * (self.dim_scale ** 3)*2)
        #print(f"After x.view: {x.shape}")
        # 将每个维度扩展4倍
        x = rearrange(x, 'b d h w (dscale hscale wscale c) -> b (d dscale) (h hscale) (w wscale) c', 
                      dscale=self.dim_scale, hscale=self.dim_scale, wscale=self.dim_scale, c=C*2)
        #print(f"After rearrange: {x.shape}")  # torch.Size([B, D*4, H*4, W*4, C*2])
        #B, D, H, W, C = x.shape
        #x = x.view(B, D * H * W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        #print(f"Final permute: {x.shape}")  # torch.Size([B, C*8, D*4, H*4, W*4])
        return x





class PTCNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, embed_dim=96, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim)
        self.encoder1 = Encoder(embed_dim, (32, 32, 32), num_heads, mlp_ratio, downsample=True)
        self.encoder2 = Encoder(embed_dim*2, (16, 16, 16), num_heads, mlp_ratio, downsample=True)
        self.encoder3 = Encoder(embed_dim*4, (8, 8, 8), num_heads, mlp_ratio, downsample=True)
        self.bottleneck = Bottleneck(embed_dim*8, (4, 4, 4), num_heads, mlp_ratio)
        self.decoder3 = Decoder(embed_dim*8, embed_dim*4, (8, 8, 8), num_heads, mlp_ratio, upsample=True)
        self.decoder2 = Decoder(embed_dim*4, embed_dim*2, (16, 16, 16), num_heads, mlp_ratio, upsample=True)
        self.decoder1 = Decoder(embed_dim*2, embed_dim, (32, 32, 32), num_heads, mlp_ratio, upsample=True)
        self.patch_expanding = PatchExpand((32, 32, 32), embed_dim, dim_scale=4)
        self.classifier = nn.Conv3d(embed_dim*2, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.patch_embedding(x)

        x1, x1_down = self.encoder1(x1)  
        #print(f"After encoder1: {x1_down.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB") #1, 192, 16, 16, 16

        x2, x2_down = self.encoder2(x1_down)
        #print(f"After encoder2: {x2_down.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")#1, 384, 8, 8, 8

        x3, x3_down = self.encoder3(x2_down)
        #print(f"After encoder3: {x3_down.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")#1, 768, 4, 4, 4

        x = self.bottleneck(x3_down)
        #print(f"After bottleneck: {x.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")#1, 768, 4, 4, 4

        x = self.decoder3(x, x3)
        #print(f"After decoder3: {x.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

        x = self.decoder2(x, x2)
        #print(f"After decoder2: {x.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")#[1, 192, 16, 16, 16]

        x = self.decoder1(x, x1)
        #print(f"After decoder1: {x.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")#1,96,32,32,32
        x = self.patch_expanding(x)
        x = self.classifier(x)
        #print(f"After classifier: {x.size()}, Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

        return x
    
    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        print("pretrained_path:", pretrained_path)
        if pretrained_path is not None:
            print("Loading pretrained model from: {}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            if "model" not in pretrained_dict:
                print("--- Loading pretrained model by splitting ---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("Deleting key: {}".format(k))
                        del pretrained_dict[k]
                self.load_state_dict(pretrained_dict, strict=False)
                return

            pretrained_dict = pretrained_dict['model']
            print("--- Loading pretrained model of encoder ---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("Deleting key: {}; shape pretrain: {}; shape model: {}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.load_state_dict(full_dict, strict=False)
        else:
            print("No pretrained model found")






if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.rand((1, 4, 128, 128, 128)).to(device)

    # model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
    #                             (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    # model(data)
    model=PTCNet().to(device)
    #print(model)
    out=model(data)
    #print("len(out):",len(out))
    #print("out1:",out[0].shape)
    print("out:",out.shape)