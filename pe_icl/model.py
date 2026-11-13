"""Single-layer Transformer for in-context linear regression with a switchable positional encoding.

The backbone is HuggingFace's GPT-2 block (transformers==4.46.3, eager attention) with two changes:
  * attention is bidirectional (the causal mask is replaced by an all-ones mask);
  * the positional encoding is one of
      - "none":    no positional information,
      - "learned": GPT-2's trainable absolute PE, initialised at `pe_scale` x the GPT-2 default (std 0.02),
      - "rope":    rotary embeddings applied to queries and keys (no trainable positional parameters).

Each prompt token is the concatenation [x_i, y_i] in R^{d+1}; the query token is [x_q, 0].
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

PE_TYPES = ("none", "learned", "rope")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len, dtype):
        return self.cos_cached[:, :, :seq_len].to(dtype), self.sin_cached[:, :, :seq_len].to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPEGPT2Attention(GPT2Attention):
    """GPT-2 eager attention with rotary embeddings applied to (head-split) queries and keys."""

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        cos, sin = self.rotary_emb(query.shape[-2], value.dtype)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return super()._attn(query, key, value, attention_mask, head_mask)


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=1024, n_layer=1, n_head=1, pe="learned", pe_scale=25.0):
        super().__init__()
        if pe not in PE_TYPES:
            raise ValueError(f"pe must be one of {PE_TYPES}, got {pe!r}")
        config = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            activation_function="gelu_new",
            attn_implementation="eager",
        )
        self.pe = pe
        self.n_dims = n_dims
        self.n_positions = n_positions
        self._read_in = nn.Linear(n_dims + 1, n_embd)
        self._backbone = GPT2Model(config)
        self._read_out = nn.Linear(n_embd, 1)

        for block in self._backbone.h:
            assert type(block.attn) is GPT2Attention, "expected eager GPT2Attention"
            block.attn.bias.fill_(True)  # bidirectional attention
            if pe == "rope":
                block.attn.__class__ = RoPEGPT2Attention
                block.attn.rotary_emb = RotaryEmbedding(block.attn.head_dim, config.max_position_embeddings)

        if pe == "learned":
            with torch.no_grad():
                self._backbone.wpe.weight.mul_(pe_scale)
        else:
            # GPT-2 always adds wpe(position_ids); zero it out so no positional signal (and no gradient) reaches it.
            self._backbone.wpe.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))

    @staticmethod
    def _combine(xs, ys):
        """Token i is [x_i, y_i]; the label of the last (query) token is zeroed out."""
        ys = ys.unsqueeze(-1).clone()
        ys[:, -1] = 0
        return torch.cat((xs, ys), dim=-1)

    def forward(self, xs, ys):
        embeds = self._read_in(self._combine(xs, ys))
        hidden = self._backbone(inputs_embeds=embeds).last_hidden_state
        return self._read_out(hidden)[:, :, 0]


def load_checkpoint(model, path, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    # Causal-mask buffers saved by older transformers versions; the mask is fixed by the constructor.
    state = {k: v for k, v in state.items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    model.load_state_dict(state)
    return model
