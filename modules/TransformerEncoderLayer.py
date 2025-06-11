import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention, LowerTriangularMask, MemoryEfficientAttentionCutlassOp
from modules.modules import apply_rotary_pos_emb, NORM_FUN, ACTIVATION_FUN


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        norm,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.norm = NORM_FUN[norm](self.hidden_size)

    def forward(
        self,
        x,
        attn_mask=None,
        kv_cache=None,
        position_embeddings=None,
    ):
        seq_len, bsz, embed_dim = x.shape
        query_states = self.q_proj(x).view(seq_len, bsz * self.num_attention_heads, self.head_dim).transpose(0, 1).contiguous()
        key_states = self.k_proj(x).view(seq_len, bsz * self.num_attention_heads, self.head_dim).transpose(0, 1).contiguous()
        value_states = self.v_proj(x).view(seq_len, bsz * self.num_attention_heads, self.head_dim).transpose(0, 1).contiguous()
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = query_states.view(bsz, self.num_attention_heads, seq_len, self.head_dim)
            key_states = key_states.view(bsz, self.num_attention_heads, seq_len, self.head_dim)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(query_states.dtype), sin.to(query_states.dtype))
            query_states = query_states.view(bsz * self.num_attention_heads, seq_len, self.head_dim)
            key_states = key_states.view(bsz * self.num_attention_heads, seq_len, self.head_dim)

        if kv_cache is not None:
            if "key_cache" in kv_cache:
                key_states = torch.cat([kv_cache["key_cache"], key_states], dim=1)
                value_states = torch.cat([kv_cache["value_cache"], value_states], dim=1)
            kv_cache["key_cache"] = key_states
            kv_cache["value_cache"] = value_states

        
        if self.training:
            attn_bias = LowerTriangularMask()
            attn = memory_efficient_attention(query_states, key_states, value_states, attn_bias, op=MemoryEfficientAttentionCutlassOp)
        else:
            query_states *= self.scaling
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )

            attn = torch.bmm(attn_weights, value_states)

        attn =  attn.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)

        attn = self.norm(attn)
        attn = self.o_proj(attn)
        return attn

class Qwen2MLP(nn.Module):
    def __init__(
        self,
            hidden_size,
            intermediate_size,
            norm,
            activation,
            **kwargs,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACTIVATION_FUN[activation]()
        self.norm = NORM_FUN[norm](intermediate_size)

    def forward(self, x):
        down_proj = self.down_proj(self.norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj

class MLP(nn.Module):
    def __init__(
        self,
            hidden_size,
            intermediate_size,
            norm,
            activation,
            **kwargs,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACTIVATION_FUN[activation]()
        self.norm = NORM_FUN[norm](intermediate_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.norm(self.act_fn(self.fc1(x)))))

class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        norm,
        activation,
        using_qwen2mlp,
    ):
        super().__init__()

        self.self_attn = Qwen2Attention(
            hidden_size,
            num_attention_heads,
            norm,
        )
        self.self_attn_norm = NORM_FUN[norm](hidden_size)

        ffn_fun = Qwen2MLP if using_qwen2mlp else MLP
        self.ffn = ffn_fun(
            hidden_size,
            hidden_size*4,
            norm,
            activation
        )
        self.ffn_norm = NORM_FUN[norm](hidden_size)

        self.dropout = nn.Dropout(0.1, inplace=True)
    def forward(
        self,
        x,
        attn_mask=None,
        kv_cache=None,
        position_embeddings=None,
    ):
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(
            x,
            attn_mask,
            kv_cache=kv_cache,
            position_embeddings=position_embeddings,
        )
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x