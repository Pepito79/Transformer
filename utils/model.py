#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)   #  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int , seq_len : int , dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        #To avoid overfitting we use a dropout layer
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,1,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(1000)/ d_model))
        #Separate the odd and even numbers
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  #(1,seq_len , d_model)
        self.register_buffer("pe",pe) #not a trainable parameter but it will be exported to the gpu 

    def forward(self ,x):
        x = x + (self.pe[: , :x.shape(1),:]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 10e-6) :
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        def forward(self,x):
            # x = (batch_size, seq_len , d_model)
            mean = x.mean(dim= -1 , keepdim= True)
            std = x.std(dim = -1 , keepdim= True)
            return self.alpha * (x - mean)/(std + eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int , d_ff: int , dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        # x : (batch_size , seq_len ,d_model) -->(batch_size , seq_len ,d_ff) -->(batch_size , seq_len ,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int , h : int , dropout: float ) -> None:
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0 ,"Can not divide d_model by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias = False )
        self.w_k= nn.Linear(d_model, d_model, bias = False )
        self.w_v = nn.Linear(d_model, d_model, bias = False )
        self.w_o = nn.Linear(d_model, d_model, bias = False )

    @staticmethod
    def attention(query,key,value,mask , dropout: nn.Dropout):
        d_k= query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.mask_fill_(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(batch_size,h,seq_len,d_model,d_model)
        if dropout is not None :
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value) , attention_scores   # Take the attention scores for visualization

    def forward(self, q, v, k , mask):
        query = self.w_q(q)  # (batch_size , seq_len , d_model ) --> (batch_size,seq_len,d_model)
        value = self.w_v(v)  # (batch_size , seq_len , d_model ) --> (batch_size,seq_len,d_model)
        key   = self.w_k(k)  # (batch_size , seq_len , d_model ) --> (batch_size,seq_len,d_model)

        #Split the embeddings
        # (batch_size , seq_len , d_model ) --> (batch_size,h,seq_len,d_model,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k ).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k ).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k ).transpose(1,2)

        x , self.attention_score = MultiHeadAttentionBlock.attention(key,query,value,mask , self.dropout)
        #(batch , h , seq_len , d_k) -->(batch_size,seq_len,h,d_k) -->(batch_size,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self,x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock ,feed_forward:FeedForwardBlock , dropout:float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self,x , src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()
    def forward(self,x,src_mask):
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)

class DecoderBlock (nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttentionBlock , feed_forward_block: FeedForwardBlock , dropout: float , cross_attention_block : MultiHeadAttentionBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range (3)])

    def forward(self,x ,encoder_output , trgt_mask , src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,trgt_mask))       
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output,src_mask))  
        x = self.residual_connections[2](x, self.feed_forward_block)       
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList , n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.layers = layers
        self.norm = LayerNormalization(n_features)
    def forward(self, x, encoder_output , src_mask , trgt_mask):
        for l in self.layers:
            x = l(x, encoder_output, src_mask, trgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int , vocab_size:int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model ,vocab_size)
    def forward(self , x) -> None:
        #(batch_size,seq_len , d_model ) -> (batch_size,seq_len,vocab_size)
        return self.projection(x)


# We want here a transformer layer that initialize with the following parameters :  
# - emb_src : InputEmbedding
# - emb_trgt: InputEmbedding
# - pe_src : PositionalEncoding
# - pe_trgt : PositionalEncoding
# - encoder : Encoder
# - decoder : Decoder
# - projection_layer : Projection_Layer

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder , decoder: Decoder , src_emb : InputEmbedding , tgt_emb: InputEmbedding , src_pe: PositionalEncoding, tgt_pe: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.src_emb = src_emb
        self.src_pe = src_pe
        self.tgt_emb = tgt_emb
        self.tgt_pe = tgt_pe
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self , input, src_mask):
        input_emb = self.src_emb(input)
        input_emb = self.src_pe(input_emb)
        return self.encoder(input_emb,src_mask)

    def decode(self , output_encoder, input_decoder, tgt_mask):
        input_decoder = self.tgt_emb(input_decoder)
        input_decoder = self.tgt_pe(input_decoder)
        return self.decoder(input_decoder, output_encoder, tgt_mask)

    def forward(self , x):
        #(batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    src_emb = InputEmbedding(vocab_size=src_vocab_size,d_model=d_model)
    tgt_emb = InputEmbedding(vocab_size=tgt_vocab_size,d_model= d_model)

    src_pe = PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=dropout)
    tgt_pe = PositionalEncoding(d_model=d_model,seq_len=tgt_seq_len,dropout=dropout)

    encoder_blocks = []
    for _ in range (N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks= []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,cross_attention_block=decoder_cross_attention_block,dropout=dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(d_model,d_ff)

    transformer = Transformer(src_emb,tgt_emb,src_pe,tgt_pe,encoder,decoder,projection_layer) 

    for p in transformer.parameters():
        if p.dim() > 1:
            #Initialize with the Glorot method
            nn.init.xavier_uniform_(p)