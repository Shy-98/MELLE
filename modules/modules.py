import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, Qwen2RMSNorm
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

NORM_FUN = {
    'rms': Qwen2RMSNorm,
    'layer': nn.LayerNorm,
}
ACTIVATION_FUN = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}
FEATURE_DIM = {
    'fbank': 80,
    'bigvgan_fbank': 80,
}

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim,
        base=10000.0,
        rope_type="default",
        max_position_embeddings=32768,
        device=None,
    ):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.base = base
        self.head_dim = head_dim
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(device=device, base=self.base, dim=self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(device=device, base=self.base, dim=self.head_dim, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def reparameterize(mu, logvar, temp=1.0):
    """
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    std = torch.exp(0.5 * logvar) * temp
    eps = torch.randn_like(std)
    return eps * std + mu

class Mel_PreNet(torch.nn.Module):
    def __init__(self, 
                 idim=80,
                 n_units=256,
                 odim=1024,
                 dropout_rate=0.5,
                 activation='relu',
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        # print(f'PreNet dropout {self.dropout_rate}')
        self.activation = ACTIVATION_FUN[activation]()

        self.prenet_l1 = nn.Linear(idim, n_units)
        self.prenet_l2 = nn.Linear(n_units, n_units)

        self.prenet_lout = nn.Linear(n_units, odim)
    
    def forward(self, x):
        x = self.activation(self.prenet_l1(x))
        x = F.dropout(x, self.dropout_rate, training=True)

        x = self.activation(self.prenet_l2(x))
        x = F.dropout(x, self.dropout_rate, training=True)

        return self.prenet_lout(x)

class Mel_PostNet(torch.nn.Module):
    def __init__(self, 
                 idim=1024,
                 odim=80,
                 n_units=256,
                 dropout_rate=0.5,
                 using_postnet=False,
                 activation='relu',
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.idim = idim
        self.odim = odim
        self.before_outs_and_logvar_l = nn.Linear(idim, odim*2)
        act_fun = ACTIVATION_FUN[activation]

        self.vae_decoder = nn.Sequential(
            nn.Linear(odim, n_units, bias=False), act_fun(), nn.Dropout(dropout_rate),
            nn.Linear(n_units, n_units, bias=False), act_fun(), nn.Dropout(dropout_rate),
            nn.Linear(n_units, odim, bias=False), nn.Dropout(dropout_rate),
        )

        if using_postnet:
            self.postnet = nn.Sequential(
                nn.Conv1d(odim, n_units, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm1d(n_units), act_fun(), nn.Dropout(dropout_rate),
                nn.Conv1d(n_units, n_units, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm1d(n_units), act_fun(), nn.Dropout(dropout_rate),
                nn.Conv1d(n_units, n_units, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm1d(n_units), act_fun(), nn.Dropout(dropout_rate),
                nn.Conv1d(n_units, n_units, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm1d(n_units), act_fun(), nn.Dropout(dropout_rate),
                nn.Conv1d(n_units, odim, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm1d(odim), nn.Dropout(dropout_rate),
            )
        else:
            self.postnet = None

    def forward(self, x):
        x = self.before_outs_and_logvar_l(x)
        mu = x[..., :self.odim]
        logvar = x[..., self.odim:]
        reparameterized_outs = reparameterize(mu, logvar)
        vae_decoder_outs = reparameterized_outs + self.vae_decoder(reparameterized_outs)

        if self.postnet is None:
            return vae_decoder_outs, mu, logvar
        else:
            outs = vae_decoder_outs + self.postnet(vae_decoder_outs.transpose(1, 2)).transpose(1, 2)
            return outs, mu, logvar, vae_decoder_outs


def make_pad_mask(lengths, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    if type(lengths) is list:
        lengths = torch.tensor(lengths)
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.shape[0]
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)