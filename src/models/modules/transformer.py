from torch import nn

from .basic_layers import MLP


class MHAttention(nn.MultiheadAttention):
    def __init__(self, d_model, num_heads, dropout=0.0, batch_first=False):
        super().__init__(d_model, num_heads, dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
        # batch_first: ``True``(N, L, d_feature) ``False``(L, N, d_feature). super().__init__(batch_first===False) as the transpose of the input is taken here.
        # self.num_heads = num_heads  # self.num_heads is already defined in nn.MultiheadAttention
        self.here_batch_first = batch_first
        self.mha_final_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        if self.here_batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if attn_mask is not None:
            batch_size = query.shape[1]
            tgt_len =  query.shape[0]
            src_len = key.shape[0]
            if attn_mask.dim() == 3 and attn_mask.shape[0] == batch_size:
                attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_heads, tgt_len, src_len).reshape(-1, tgt_len, src_len)
        
        attn_output, attn_output_weights = super().forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
            )
        
        if self.here_batch_first:
            attn_output = attn_output.transpose(1, 0)
    
        return self.mha_final_dropout(attn_output)
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.0, batch_first=False, norm_first=False):
        super().__init__()
        
        self.self_attn = MHAttention(d_model, num_heads, dropout, batch_first)
        self.ffn = MLP(d_model, [d_ffn, d_model], nn.SiLU, dropout, trunc_normal_init=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        
    def _forward_self_attn(self, x, key_padding_mask, need_weights, attn_mask):
        return self.self_attn(x, x, x, key_padding_mask, need_weights, attn_mask)
        
    def forward(self, x, key_padding_mask=None, need_weights=False, attn_mask=None):
        if self.norm_first:
            x = x + self._forward_self_attn(self.norm1(x), key_padding_mask, need_weights, attn_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self._forward_self_attn(x, key_padding_mask, need_weights, attn_mask))
            x = self.norm2(x + self.ffn(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.0, batch_first=False, norm_first=False):
        super().__init__()
        
        self.self_attn = MHAttention(d_model, num_heads, dropout, batch_first)
        self.cross_attn = MHAttention(d_model, num_heads, dropout, batch_first)
        self.ffn = MLP(d_model, [d_ffn, d_model], nn.SiLU, dropout, trunc_normal_init=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        
    def _forward_self_attn(self, x, self_key_padding_mask, self_need_weights, self_attn_mask):
        return self.self_attn(x, x, x, self_key_padding_mask, self_need_weights, self_attn_mask)
    
    def _forward_cross_attn(self, x, momery, cross_key_padding_mask, cross_need_weights, cross_attn_mask):
        return self.cross_attn(x, momery, momery, cross_key_padding_mask, cross_need_weights, cross_attn_mask)
    
    def forward(self, x, memory, self_key_padding_mask=None, cross_key_padding_mask=None, self_need_weights=False, cross_need_weights=False, self_attn_mask=None, cross_attn_mask=None):
        if self.norm_first:
            x = x + self._forward_self_attn(self.norm1(x), self_key_padding_mask, self_need_weights, self_attn_mask)
            x = x + self._forward_cross_attn(self.norm2(x), memory, cross_key_padding_mask, cross_need_weights, cross_attn_mask)
            x = x + self.ffn(self.norm3(x))
        else:
            x = self.norm1(x + self._forward_self_attn(x, self_key_padding_mask, self_need_weights, self_attn_mask))
            x = self.norm2(x + self._forward_cross_attn(x, memory, cross_key_padding_mask, cross_need_weights, cross_attn_mask))
            x = self.norm3(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn=None, dropout=0.0, batch_first=False, norm_first=False, norm_out=True):
        super().__init__()
        
        d_ffn = d_ffn if d_ffn is not None else 4 * d_model
        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ffn, dropout, batch_first, norm_first) for _ in range(num_layers)]
            )
        self.norm = nn.LayerNorm(d_model) if norm_out else nn.Identity()
    
    def forward(self, x, key_padding_mask=None, need_weights=False, attn_mask=None):
        for transformer_encoder_layer in self.transformer_encoder_layers:
            x = transformer_encoder_layer(x, key_padding_mask, need_weights, attn_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn=None, dropout=0.0, batch_first=False, norm_first=False, norm_out=True):
        super().__init__()
        
        d_ffn = d_ffn if d_ffn is not None else 4 * d_model
        self.transformer_decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ffn, dropout, batch_first, norm_first) for _ in range(num_layers)]
            )
        self.norm = nn.LayerNorm(d_model) if norm_out else nn.Identity()
        
    def forward(self, x, memory, self_key_padding_mask=None, cross_key_padding_mask=None, self_need_weights=False, cross_need_weights=False, self_attn_mask=None, cross_attn_mask=None):
        for transformer_decoder_layer in self.transformer_decoder_layers:
            x = transformer_decoder_layer(x, memory, self_key_padding_mask, cross_key_padding_mask, self_need_weights, cross_need_weights, self_attn_mask, cross_attn_mask)
        return self.norm(x)
        

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ffn=None, dropout=0.0, batch_first=False, norm_first=False, norm_out=True):
        super().__init__()
        
        d_ffn = d_ffn if d_ffn is not None else 4 * d_model
        self.transformer_encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ffn, dropout, batch_first, norm_first, norm_out)
        self.transformer_decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ffn, dropout, batch_first, norm_first, norm_out)

    def forward(self, src, tgt, encoder_key_padding_mask=None, encoder_need_weights=False, encoder_attn_mask=None,
                self_key_padding_mask=None, cross_key_padding_mask=None, self_need_weights=False, cross_need_weights=False, self_attn_mask=None, cross_attn_mask=None):
        memory = self.transformer_encoder(src, encoder_key_padding_mask, encoder_need_weights, encoder_attn_mask)
        tgt = self.transformer_decoder(tgt, memory, self_key_padding_mask, cross_key_padding_mask, self_need_weights, cross_need_weights, self_attn_mask, cross_attn_mask)
        return tgt
