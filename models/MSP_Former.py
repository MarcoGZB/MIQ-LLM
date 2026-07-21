import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

class FaCBModule(nn.Module):
    """
    FaCB (Feature and Class Balance) 
    """
    def __init__(self, d_model=512):
        super(FaCBModule, self).__init__()
        self.balance_weights = nn.Parameter(torch.ones(1))

    def forward(self, features, labels=None):
        norm_features = F.normalize(features, p=2, dim=-1)
        # FaCB loss ...
        loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss

class MSPSelfAttention(nn.Module):
    """
    Self-Attention
    """
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 512 // 8 = 64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MSPMLP(nn.Module):
    """
    MLP (FFN) 模块
    """
    def __init__(self, in_features=512, hidden_features=2048, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MSPEncoderBlock(nn.Module):
    """
    MSP-Former
    """
    def __init__(self, dim=512, num_heads=8, mlp_hidden=2048, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSPSelfAttention(dim=dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSPMLP(in_features=dim, hidden_features=mlp_hidden, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class FeatureTokenFusion:
    """
    Feature Token Fusion
    """
    @staticmethod
    def fuse(ts_features, inputs_embeds, input_ids, pad_token_id=151643):

        batch_size, seq_len, embed_dim = inputs_embeds.shape
        num_tss, num_ts_patches, ts_embed_dim = ts_features.shape
        
        assert embed_dim == ts_embed_dim, f"Embedding dimensions mismatch: LLM {embed_dim} vs TS {ts_embed_dim}"

        batch_indices, seq_indices = torch.where(input_ids == pad_token_id)
        
        ts_features_flat = ts_features.view(-1, embed_dim).to(
            dtype=inputs_embeds.dtype, 
            device=inputs_embeds.device
        )
        
        # In-place tensor replacement (clone first to avoid leaf variable inplace issues during autograd)
        inputs_embeds = inputs_embeds.clone()
        replace_len = min(len(batch_indices), num_tss * num_ts_patches)
        
        inputs_embeds[batch_indices[0:replace_len], seq_indices[0:replace_len]] = ts_features_flat[0:replace_len]
        
        return inputs_embeds

class MSPFormer(nn.Module):
    """
    MSP-Former
    """
    def __init__(self, d_model=512, e_layers=4, n_heads=8, llm_d_model=4096):
        super(MSPFormer, self).__init__()
        

        self.encoder_layers = nn.ModuleList([
            MSPEncoderBlock(dim=d_model, num_heads=n_heads, mlp_hidden=d_model*4) 
            for _ in range(e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        self.token_reprogram = nn.Linear(d_model, llm_d_model)

        self.facb_loss_module = FaCBModule(d_model=d_model)

    def forward(self, ts_tokens, labels=None, attn_mask=None):
        """
        ts_tokens: [B, L, d_model=512]
        """
        # ---  ---
        x = ts_tokens
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
        x = self.encoder_norm(x)
        
        # --- (FaCB) ---
        facb_loss = None
        if self.training:
            facb_loss = self.facb_loss_module(x, labels)
            
        # --- Token Reprogram  ---
        aligned_features = self.token_reprogram(x) # [B, L, 4096]
        
        return aligned_features, facb_loss