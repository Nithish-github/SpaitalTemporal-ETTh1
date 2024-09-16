import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from IPython import embed

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 

                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 

                output_attention = False, distil=True, mix=True,

                device=torch.device('cuda:0')):
        
        super(Informer, self).__init__()

        self.pred_len = out_len

        self.attn = attn


        self.label_len = label_len    #storing the label length



        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, seq_len, d_model, embed, freq, dropout)

        self.dec_embedding = DataEmbedding(dec_in, label_len + out_len, d_model, embed, freq, dropout)



        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model, seq_len
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc) #enc_out:(32,seq_len,512)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #enc_out:(32,pred_len,512) 
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # Autoregressive decoding loop for generating additional 90 tokens
        for i in range(self.label_len-1):
            # print("Inside loop")
            # Pass current decoder output (dec_out) and encoder context (enc_out)
            dec_out_step = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            # Take the last predicted token and use it for the next input step
            dec_out = torch.cat([dec_out, dec_out_step[:, -self.label_len:, :]], dim=1)  # Concatenate the new token [batch_size, current_len + 1, d_model]


        # Once the full sequence is generated, project it to the desired output dimension (e.g., 1)
        final_out = self.projection(dec_out)  # Project to [batch_size, pred_len, c_out]


        if self.output_attention:
            return final_out, attns
        else:
            return final_out # [B, L, D] #(batch,pred_len,1)
        

