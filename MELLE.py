import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.TransformerEncoderLayer import EncoderLayer
from modules.modules import Mel_PreNet, Mel_PostNet, make_pad_mask, NORM_FUN, ACTIVATION_FUN, Qwen2RotaryEmbedding

logger = logging.getLogger(__name__)

def init_bert_params(module, scaling):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).mul_(scaling).to(data.device))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class MELLE(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_attention_heads=16,
        encoder_layers=12,
        feature_dim=80,
        using_rope=False,
        using_postnet=True,
        using_qwen2mlp=True,
        norm='rms', # [rms, layer]
        transformer_activation='silu', # [silu relu]
        prenet_activation='silu', # [silu relu]
        postnet_activation='silu', # [silu relu]
        ):
        super().__init__()
        norm_fun = NORM_FUN[norm]
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.using_rope = using_rope
        self.using_postnet = using_postnet

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(
                hidden_size, num_attention_heads, 
                norm, transformer_activation,
                using_qwen2mlp
            ) for _ in range(encoder_layers)]
        )

        self.text_embedding = nn.Embedding(32000, hidden_size, padding_idx=0) # LlamaTokenizerFast
        # self.text_embedding = nn.Embedding(151643+2, hidden_size, padding_idx=0) # Qwen2TokenizerFast
        self.mel_embedding = Mel_PreNet(idim=self.feature_dim, activation=prenet_activation)

        if using_rope:
            self.rotary_emb = Qwen2RotaryEmbedding(head_dim=hidden_size // num_attention_heads)
        else:
            self.mel_position = nn.Embedding(5000, hidden_size)
            self.text_position = nn.Embedding(5000, hidden_size)

        self.mel_decoder = Mel_PostNet(odim=self.feature_dim, using_postnet=using_postnet, activation=postnet_activation)

        self.stop_projection = nn.Linear(hidden_size, 1)
        self.mel_bos_embed = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.normal_(self.mel_bos_embed, mean=0, std=hidden_size**-0.5)

        self.dropout = nn.Dropout(0.1, inplace=True)
        self.layer_norm = norm_fun(hidden_size)
        self.apply(partial(init_bert_params, scaling=math.sqrt(math.log(encoder_layers * 2))))


    def forward(
          self,
          mel=None,
          mel_lengths=None,
          txt=None,
          txt_lengths=None,
          *args,
          **kwargs,
        ):

        mel_max_length = mel_lengths.max()
        txt_max_length = txt_lengths.max()
        mel_mask = ~make_pad_mask(mel_lengths)
        txt_mask = ~make_pad_mask(txt_lengths)

        txt_emb = self.text_embedding(txt)
        mel_emb = self.mel_embedding(mel[:, :-1])
        mel_emb = torch.cat([self.mel_bos_embed.expand(mel_emb.shape[0], -1, -1), mel_emb], dim=1)
        
        src = torch.zeros((
            mel_emb.shape[0], txt_max_length+mel_max_length, mel_emb.shape[2]
        ), dtype=mel_emb.dtype, device=mel_emb.device)

        if self.using_rope:
            txt_mel_mask = torch.cat([txt_mask, mel_mask], dim=1)
            position_ids = torch.cumsum(txt_mel_mask, dim=-1)
            position_embeddings = self.rotary_emb(src, position_ids)
            position_embeddings[0][~txt_mel_mask] = 0.
            position_embeddings[1][~txt_mel_mask] = 0.
        else:
            mel_position_ids = torch.cumsum(mel_mask, dim=-1)
            txt_position_ids = torch.cumsum(txt_mask, dim=-1)
            txt_position_emb = self.text_position(txt_position_ids)
            txt_emb = txt_emb + txt_position_emb
            mel_position_emb = self.mel_position(mel_position_ids)
            mel_emb = mel_emb + mel_position_emb
            position_embeddings = None

        
        src[:, :txt_max_length][txt_mask] = txt_emb[txt_mask]
        src[:, txt_max_length:][mel_mask] = mel_emb[mel_mask]
        
        src = self.dropout(src)

        src = src.transpose(0,1)
        attn_mask=torch.triu(
            torch.zeros([src.shape[0], src.shape[0]], dtype=src.dtype, device=src.device).fill_(float("-inf")),
            1,
        )
        for idx, layer in enumerate(self.encoder_layers):
            src = layer(
                src, # causal mask will create in Attention module
                position_embeddings=position_embeddings,
                attn_mask=attn_mask
            )
        src = self.layer_norm(src)
        encoder_out = src.transpose(0,1)
        encoder_out = encoder_out[:, txt_max_length:]
        stop_logits = self.stop_projection(encoder_out)

        if self.using_postnet:
            outs, mu, logvar, vae_decoder_outs = self.mel_decoder(encoder_out)
        else:
            vae_decoder_outs, mu, logvar = self.mel_decoder(encoder_out)

        target_choice = ~make_pad_mask(mel_lengths-1, max_len=mel_max_length).unsqueeze(-1)
        mel_target = mel.masked_select(target_choice).view(-1)

        mu = mu.masked_select(target_choice).view(-1)
        logvar = logvar.masked_select(target_choice).view(-1)
        vae_decoder_outs = vae_decoder_outs.masked_select(target_choice).view(-1)
        
        spec_flux_for_loss = F.l1_loss(mu.view(-1,self.feature_dim)[1:], mel_target.view(-1,self.feature_dim)[:-1], reduction='sum')
        loss_l1 = F.l1_loss(vae_decoder_outs, mel_target, reduction='sum') 
        loss_l2 = F.mse_loss(vae_decoder_outs, mel_target, reduction='sum') 

        if self.using_postnet:
            outs = outs.masked_select(target_choice).view(-1)
            loss_l1 = loss_l1 + F.l1_loss(outs, mel_target, reduction='sum')
            loss_l2 = loss_l2 + F.mse_loss(outs, mel_target, reduction='sum')
        else:
            loss_l1 = 2*loss_l1
            loss_l2 = 2*loss_l2

        loss_logvar = (- (1 + logvar - (mu-mel_target).pow(2) - logvar.exp())).sum()
        
        stop_choice = mel_mask.unsqueeze(-1)
        stop_target = (~target_choice).type_as(stop_logits).masked_select(stop_choice).view(-1)
        stop_logits = stop_logits.masked_select(stop_choice).view(-1)
        loss_bce = F.binary_cross_entropy_with_logits(stop_logits, stop_target, pos_weight=torch.tensor(100.0), reduction='sum')

        loss = loss_l1 + loss_l2 + 5e-2 * loss_logvar - 1.0 * spec_flux_for_loss + loss_bce

        return loss, loss_l1, loss_l2, loss_logvar, loss_bce
    
    def init_kv_cache(
        self,
        mel=None,
        txt=None,
        kv_cache=None,
    ):
        txt_emb = self.text_embedding(txt)
        mel_emb = self.mel_embedding(mel[:, :-1])
        mel_emb = torch.cat([self.mel_bos_embed.expand(mel_emb.shape[0], -1, -1), mel_emb], dim=1)
        src = torch.cat([txt_emb, mel_emb], dim=1)

        if self.using_rope:
            position_ids = torch.range(1, txt.shape[1]+mel.shape[1], dtype=torch.long, device=src.device).reshape(1,-1)
            position_embeddings = self.rotary_emb(src, position_ids)
        else:
            txt_position_ids = torch.range(1, txt_emb.shape[1], dtype=torch.long, device=txt_emb.device).reshape(1,-1)
            mel_position_ids = torch.range(1, mel_emb.shape[1], dtype=torch.long, device=mel_emb.device).reshape(1,-1)
            txt_position_emb = self.text_position(txt_position_ids)
            mel_position_emb = self.mel_position(mel_position_ids)
            txt_pos_emb = txt_emb+txt_position_emb
            mel_pos_emb = mel_emb+mel_position_emb
            src = torch.cat([txt_pos_emb, mel_pos_emb], dim=1)
            position_embeddings = None
        
        src = self.dropout(src)

        src = src.transpose(0,1)
        attn_mask=torch.triu(
            torch.zeros([src.shape[0], src.shape[0]], dtype=src.dtype, device=src.device).fill_(float("-inf")),
            1,
        )
        for idx, layer in enumerate(self.encoder_layers):
            src = layer(
                src,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache[idx],
                attn_mask=attn_mask
            )
        
        return None

    @torch.no_grad()
    def inference(
        self,
        mel=None,
        txt=None,
        max_length=1000,
    ):
        orig_mel_length = mel.shape[1]
        kv_cache = [{} for _ in range(len(self.encoder_layers))]
        self.init_kv_cache(mel, txt, kv_cache)

        while True:
            current_mel = mel[:,-1:]
            src = self.mel_embedding(current_mel)
            position_ids = torch.tensor(txt.shape[1]+mel.shape[1]+1, dtype=torch.long, device=src.device).reshape(1,-1)
            if self.using_rope:
                position_embeddings = self.rotary_emb(src, position_ids)
            else:
                position_embeddings = self.mel_position(position_ids)
                src = src + position_embeddings
                position_embeddings = None
            src = self.dropout(src)

            src = src.transpose(0,1)
            for idx, layer in enumerate(self.encoder_layers):
                src = layer(
                    src, # causal mask will create in Attention module
                    position_embeddings=position_embeddings,
                    kv_cache=kv_cache[idx],
                )
            src = src.transpose(0,1)
            encoder_out = self.layer_norm(src)
            stop_logits = self.stop_projection(encoder_out)
            if self.using_postnet:
                _, _, _, vae_decoder_outs = self.mel_decoder(encoder_out)
            else:
                vae_decoder_outs, _, _ = self.mel_decoder(encoder_out)
            mel = torch.cat([mel, vae_decoder_outs], dim=1)
            if (stop_logits[0][0] > 0. or mel.shape[1] >= max_length): break

        if self.using_postnet:
            mel[:,orig_mel_length:] += self.mel_decoder.postnet(mel[:,orig_mel_length:].transpose(1,2)).transpose(1,2)
        print(f'{orig_mel_length} --> {mel.shape[1]}')
        return mel[:,orig_mel_length:]
        

