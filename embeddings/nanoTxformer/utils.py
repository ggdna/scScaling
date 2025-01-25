import torch

def get_mean_pooled_embeddings(model, adata, chunk_size=10, layer_index=None):
    """
    Get mean-pooled embeddings for the entire dataset in chunks.
    
    Args:
    - model (nn.Module): The transformer model with a mean_pooling method.
    - dataset (torch.Tensor): The dataset of tokenized sequences (shape: total_samples, seq_len).
    - chunk_size (int): The number of sequences in each chunk (batch size).
    
    Returns:
    - all_mean_pooled (torch.Tensor): The mean-pooled embeddings for the entire dataset.
    """

    model.eval()

    tokens = model.tokenizer.tokenize_adata(adata)
    tokenized_data = torch.tensor(tokens, dtype=torch.long).cuda()
    
    all_mean_pooled = []
    
    total_samples = tokenized_data.shape[0]
    num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Compute number of chunks

    with torch.no_grad():  
        for i in range(num_chunks):
            # Get the chunk of data
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            batch = tokenized_data[start_idx:end_idx]

            # Compute the mean-pooled embedding for the batch
            mean_pooled = model.mean_pooling(batch, layer_index=layer_index)  # Shape: (batch_size, embed_size)

            # Append the results
            all_mean_pooled.append(mean_pooled)

    # Concatenate all the mean-pooled embeddings
    all_mean_pooled = torch.cat(all_mean_pooled, dim=0)  # Shape: (total_samples, embed_size)

    return all_mean_pooled