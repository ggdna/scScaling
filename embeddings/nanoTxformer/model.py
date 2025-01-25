import torch
import torch.nn as nn
from . import tokenizer

class nanoTxformer(nn.Module):
    def __init__(self, adata,  
                 embed_size=128, 
                 num_heads=2, 
                 num_encoder_layers=2, 
                 num_decoder_layers=0, 
                 forward_expansion=4, 
                 dropout=0.1,):
        super(nanoTxformer, self).__init__()
        

        num_tokens = len(adata.var)+2
        max_length = len(adata.var)
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_size))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_encoder_layers
        )
                
        self.fc_out = nn.Linear(embed_size, num_tokens)
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = tokenizer.scTokenizer(adata)

    def forward(self, src):
        # Embed and add positional encoding to the source and target sequences
        src_seq_len = src.shape[1]
        src = self.dropout(self.embedding(src) + self.positional_encoding[:, :src_seq_len, :])
        output = self.transformer(src)
        out = self.fc_out(output)

        return out
    
    def mean_pooling(self, src, layer_index=None):
        """
        compute the mean-pooled embedding from a specified encoder layer.

        Args:
        - src (torch.Tensor): input sequence of tokens, shape (batch_size, seq_len).
        - layer_index (int, optional): index of the encoder layer to extract embeddings from. 
        defaults to the last layer if None.

        Returns:
        - mean_pooled (torch.Tensor): mean-pooled embedding of shape (batch_size, embed_size).
        """
        # embed and add positional encoding to the source sequence
        src_seq_len = src.shape[1]
        src = self.dropout(self.embedding(src) + self.positional_encoding[:, :src_seq_len, :])

        # pass through encoder
        all_encoder_outputs = []  # to store outputs of all layers
        x = src
        for layer in self.transformer.layers:
            x = layer(x)
            all_encoder_outputs.append(x)

        # if layer_index is None, use the last layer
        if layer_index is None:
            layer_index = len(all_encoder_outputs) - 1

        # handle invalid layer index
        if layer_index < 0 or layer_index >= len(all_encoder_outputs):
            raise ValueError(f"layer_index {layer_index} is out of range. must be between 0 and {len(all_encoder_outputs) - 1}.")

        # get embeddings from the specified layer and mean-pool over sequence dimension
        selected_output = all_encoder_outputs[layer_index]
        mean_pooled = selected_output.mean(dim=1)

        return mean_pooled
