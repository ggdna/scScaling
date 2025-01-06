from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

MASK_PROB = 0.15  # Probability of masking a token

def mask_tokens(input_sequences, mask_token, mask_prob=MASK_PROB):
    # Create a copy of the input to be the target
    targets = input_sequences.clone()

    # Randomly mask tokens with a probability
    mask = torch.bernoulli(torch.full(input_sequences.shape, mask_prob)).bool()

    # Apply the mask to the input (replace some tokens with the MASK_TOKEN)
    input_sequences[mask] = mask_token  # Replace the selected tokens with the [MASK] token

    return input_sequences, targets, mask

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a learning rate scheduler with warm-up followed by linear decay.
    
    Args:
    - optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for.
    - num_warmup_steps (int): The number of steps to linearly increase the learning rate.
    - num_training_steps (int): The total number of training steps.
    
    Returns:
    - scheduler (torch.optim.lr_scheduler.LambdaLR): A learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warm-up phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay phase
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopper:
    """
    early stopping that returns best weights
    trying to replicate the Keras callback
    """
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state = None

    def early_stop(self, validation_loss, model):
        """
        checks if training should be stopped based on validation loss and patience
        
        :param validation_loss: current validation loss
        :param model: the model being trained
        :return: best model weights if early stopping criteria is met, else False
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return self.best_state
        return False
    
def generalization_loss(model, adata, batch_size=1):
    """
    computes the (generalization) loss on a dataset
    
    :param model: the model to evaluate
    :param adata: anndata object
    """

    mask_token = model.num_tokens - 1
    criterion = nn.CrossEntropyLoss()
    
    tokens = model.tokenizer.tokenize_adata(adata)
    tokenized_data = torch.tensor(tokens, dtype=torch.long).cuda()
    val_loader = DataLoader(tokenized_data, batch_size=1, shuffle=False)
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for src in val_loader:
            src_masked, targets, mask = mask_tokens(src, mask_token)
            output = model(src_masked)
            
            output = output.view(-1, output.shape[-1])
            targets = targets.view(-1)
            mask = mask.view(-1)
            
            output_masked = output[mask]
            targets_masked = targets[mask]
            
            val_loss = criterion(output_masked, targets_masked)
            epoch_val_loss += val_loss.item()
    
    return epoch_val_loss / len(val_loader)
    

def train_model(model, adata, epochs, patience=1, 
                validation_split=0.2, val_per_batch=10**3, batch_size=32):
    """
    trains the model on tokenized data, tracks training and validation loss, and applies early stopping.
    
    :param model: the model to train
    :param adata: AnnData object containing the single-cell data
    :param epochs: total number of epochs to train
    :param patience: number of epochs to wait for improvement in validation loss before stopping early
    :param validation_split: fraction of data to use for validation
    :param val_per_batch: number of batches in between validation steps
    :return: lists of training and validation losses for each epoch, and best model weights
    """
    
    # Tokenize and convert the data to tensors
    tokens = model.tokenizer.tokenize_adata(adata)
    tokenized_data = torch.tensor(tokens, dtype=torch.long).cuda()
    
    # Train-test split
    dataset_size = len(tokenized_data)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    train_data, val_data = random_split(tokenized_data, [train_size, val_size])
    
    # Data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    mask_token = model.num_tokens - 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler with warmup and linear decay
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    
    # Track losses and early stopping
    train_losses, val_losses = [], []
    early_stopper = EarlyStopper(patience=patience)
    
    model.train()
    cumulative_batch_count = 0  # Track total number of batches across epochs

    for epoch in range(epochs):
        epoch_train_loss = 0
        for batch_idx, src in enumerate(train_loader):
            src_masked, targets, mask = mask_tokens(src, mask_token)
            
            
            optimizer.zero_grad()
            
            
            output = model(src_masked)
            output = output.view(-1, output.shape[-1])
            targets = targets.view(-1)
            
            # Only calculate loss on masked positions
            mask = mask.view(-1)
            output_masked = output[mask]
            targets_masked = targets[mask]
            
            loss = criterion(output_masked, targets_masked)
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()            
            epoch_train_loss += loss.item()
            cumulative_batch_count += 1 
            
            if cumulative_batch_count % val_per_batch == 0:
                # Validation step
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for src in val_loader:
                        src_masked, targets, mask = mask_tokens(src, mask_token)
                        output = model(src_masked)
                        
                        output = output.view(-1, output.shape[-1])
                        targets = targets.view(-1)
                        mask = mask.view(-1)
                        
                        output_masked = output[mask]
                        targets_masked = targets[mask]
                        
                        val_loss = criterion(output_masked, targets_masked)
                        epoch_val_loss += val_loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f'epoch {epoch + 1}/{epochs} (batch {cumulative_batch_count}) - train loss: {epoch_train_loss / (batch_idx + 1):.4f}, val loss: {avg_val_loss:.4f}')
                
                best_weights = early_stopper.early_stop(avg_val_loss, model)

                if best_weights:
                    print(f"Early stopping triggered at epoch {epoch + 1}, batch {cumulative_batch_count}")
                    model.load_state_dict(best_weights)
                    return train_losses, val_losses
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
    
    return train_losses, val_losses
