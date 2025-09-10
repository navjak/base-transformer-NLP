import torch
import torch.nn as nn
import torchvision
import math


# Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # paper mentions they divided it by sqrt of d_model
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # matrix with shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # vector of shape (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # numerator term of the positional encoding formulas
        denom_term = torch.exp(torch. arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator term of the positional encoding formulas

        # sin for even positions
        pe[:, 0::2] = torch.sin(pos * denom_term)

        # cos for odd positions
        pe[:, 1::2] = torch.cos(pos * denom_term)

        # have to add batch dimension to pe tensor
        pe = pe.unsqeeze(0) # shape (1, seq_len, d_model)
        self.register_buffer('pe', pe) # register pe as a buffer so that it is not considered a model parameter but still gets saved and loaded with the model

    def forward(self, x):
        x = x + (self.pe[:, :x_shape[1], :]).requires_grad_(False) # pe is not a model parameter so we don't want to compute gradients for it
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied || gain parameter
        self.bias = nn.Parameter(torch.zeros(1)) # added || bias parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std_dev = x.std(dim = -1, keepdim=True)
        normalized_x = (x - mean) / (std_dev + self.eps)
        return self.alpha * normalized_x + self.bias


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, df_model: int, num_heads: int, dropout: float):
        super().__init__()     
        self.num_heads = num_heads
        self.d_model = d_model   

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod # allows calling this function without an instance of the class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]   
        attn_scores =  (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask (for Masked Multi-Head Attention in the decoder)
        if mask is not None:
            attn_scores.masked_fill(mask == 0, float('-inf'))

        # softmax
        attn_scores = attn_scores.softmax(dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # dropout
        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ value), attn_scores


    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # # (batch_size, seq_len, d_model ) --> (batch_size, num_heads, seq_len, d_k )
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len, d_k )
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        
        att_x, attn_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # att_x shape (batch_size, num_heads, seq_len, d_k )

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1,self.num_heads * self.d_model) # (batch_size, seq_len, d_model)

        x = self.w_o(x) # (batch_size, seq_len, d_model)
        return x
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        norm_x = self.norm(x)
        return x + self.dropout(sublayer(self.norm_x)) # residual connection + dropout


# Single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, ffn_block: FFN, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ffn_block = ffn_block
        self.enc_residual1 = ResidualConnection(dropout)
        self.enc_residual2 = ResidualConnection(dropout)

    def forward(self, x, src_mask):
        x = self.enc_residual1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.enc_residual2(x, self.ffn_block)
        return x

# multiple encoder layers make Encoder
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x , mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

# Single Decoder layer/block 
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, ffn_block: FFN, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ffn_block = ffn_block
        self.dec_residual1 = ResidualConnection(dropout) 
        self.dec_residual2 = ResidualConnection(dropout)    
        self.dec_residual3 = ResidualConnection(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.dec_residual1(x, lambda x:self.self_attention_block(x, x, x, tgt_mask))
        x = self.dec_residual2(x, lambda x: self.cross_attention_block(x, enc_output, enc_output,src_mask))
        x = self.dec_residual3(x, self.ffn_block)
        return x

# multiple decoder layers make Decoder
class Decoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm - LayerNormalization(features)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)
    

# Main Transofrmer class
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


# function to build model
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # initialize params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

