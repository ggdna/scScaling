import torch
import torch.nn as nn
from . import tokenizer

class nanoTxformer(nn.Module):
    def __init__(self, adata,  
                 embed_size=32, 
                 num_heads=1, 
                 num_encoder_layers=3, 
                 num_decoder_layers=0, 
                 forward_expansion=4, 
                 dropout=0.1,):
        super(nanoTxformer, self).__init__()
        

        num_tokens = len(adata.var)+2
        max_length = len(adata.var)
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_size))

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embed_size, num_tokens)
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = tokenizer.scTokenizer(adata)

    def forward(self, src):
        # Embed and add positional encoding to the source and target sequences
        src_seq_len = src.shape[1]
        src = self.dropout(self.embedding(src) + self.positional_encoding[:, :src_seq_len, :])
        output = self.transformer(src, src)
        out = self.fc_out(output)

        return out
    
    def mean_pooling(self, src):
        """
        compute the mean-pooled embedding
        
        Args:
        - src (torch.Tensor): Input sequence of tokens, shape (batch_size, seq_len).
        
        Returns:
        - mean_pooled (torch.Tensor): Mean-pooled embedding of shape (batch_size, embed_size).
        """
        # Embed and add positional encoding to the source sequences
        src_seq_len = src.shape[1]
        src = self.dropout(self.embedding(src) + self.positional_encoding[:, :src_seq_len, :])

        transformer_output = self.transformer(src, src)

        # mean pooling over the sequence dimension (dim=1)
        mean_pooled = transformer_output.mean(dim=1)
                
        return mean_pooled
