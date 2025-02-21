import math
import random
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim

##############################
# Gradient Clipping Utility  #
##############################

class GradientClipping:
    def __init__(self, clip_value):
        self.epoch_grads = []
        self.total_grads = []
        self.clip = clip_value

    def track_grads(self, module, grad_input, grad_output):
        # Track norm of the first gradient in the tuple (if available)
        if grad_input[0] is not None:
            grad_norm = grad_input[0].detach().norm().item()
            self.epoch_grads.append(grad_norm)

    def register_hook(self, module):
        # Use full backward hook (available in recent PyTorch versions)
        module.register_full_backward_hook(self.track_grads)

    def gradient_mean(self):
        return np.mean(self.epoch_grads) if self.epoch_grads else 0.0

    def gradient_std(self):
        return np.std(self.epoch_grads) if self.epoch_grads else 0.0

    def reset_gradients(self):
        self.total_grads.extend(self.epoch_grads)
        self.epoch_grads = []

    def update_clip_value(self):
        self.clip = self.gradient_mean() + self.gradient_std()

    def update_clip_value_total(self):
        all_grads = self.total_grads + self.epoch_grads
        self.clip = np.mean(all_grads) if all_grads else self.clip

######################################
# Transformer Components and Layers  #
######################################

class FF(nn.Module):
    """Point-wise Feed-Forward Network used in transformer layers."""
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Linear projections
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # Split into heads: shape -> (batch_size, n_heads, seq_len, d_head)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            # Expand mask for all heads and set masked positions to a very negative value
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head)
        out = self.fc(context)
        return out

class EncoderCell(nn.Module):
    """A single transformer encoder cell with self-attention and feed-forward network."""
    def __init__(self, d_model, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.ff = FF(d_model, hidden_size, dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention sub-layer with residual connection and layer normalization
        attn_out = self.self_attn(x, x, x, mask)
        x = self.lnorm1(x + self.dropout(attn_out))
        # Feed-forward sub-layer with residual connection and layer normalization
        ff_out = self.ff(x)
        x = self.lnorm2(x + self.dropout(ff_out))
        return x

class Encoder(nn.Module):
    """Stack of transformer encoder cells."""
    def __init__(self, d_model, hidden_size, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderCell(d_model, hidden_size, n_heads, dropout) for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

#################################
# BSTransformer Model           #
#################################

class BSTransformer(nn.Module):
    """
    Behavior Sequence Transformer model that:
      - Embeds user behavior sequences and other features
      - Uses a transformer encoder to capture sequential relations
      - Concatenates aggregated transformer output with context embeddings
      - Applies an MLP to output logits for all items (for CTR prediction)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding layer for sequence items (e.g. item_id)
        self.item_embed = nn.Embedding(
            num_embeddings=config['item_embed']['num_embeddings'],
            embedding_dim=config['item_embed']['embedding_dim'],
            padding_idx=config['item_embed']['padding_idx']
        )
        # Positional embedding (using sinusoidal encoding here)
        if config.get('use_sinusoidal_pos', True):
            pos_emb = self.sinusoidal_pos_embedding(
                config['max_seq_len'], config['item_embed']['embedding_dim']
            )
            # Register as a buffer so it’s part of the model but not updated
            self.register_buffer('pos_embedding_buffer', pos_emb)
        else:
            self.pos_embed = nn.Embedding(config['max_seq_len'], config['item_embed']['embedding_dim'])
        
        # Embeddings for "other" features (e.g. user profile, context)
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=feat['num_embeddings'],
                embedding_dim=feat['embedding_dim'],
                padding_idx=feat['padding_idx']
            ) for feat in config['context_features']
        ])
        
        # Transformer encoder (self-attention layer)
        self.encoder = Encoder(
            d_model=config['trans']['input_size'],
            hidden_size=config['trans']['hidden_size'],
            n_layers=config['trans']['n_layers'],
            n_heads=config['trans']['n_heads'],
            dropout=config.get('dropout', 0.1)
        )
        
        # MLP layers: concatenates the transformer output with context embeddings,
        # then applies several fully-connected layers.
        mlp_input_size = config['trans']['input_size'] + sum(
            [feat['embedding_dim'] for feat in config['context_features']]
        )
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(512, config['item_embed']['num_embeddings'])
        )
        
        # Parameter initialization
        for param in self.parameters():
            if param.dim() > 1:
                if config.get('init_method', 'xavier') == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif config.get('init_method') == 'kaiming':
                    nn.init.kaiming_uniform_(param)
        print(f"Parameters initialized using {config.get('init_method', 'xavier')} initialization!")
    
    def sinusoidal_pos_embedding(self, max_seq_len, embedding_dim):
        """Creates a sinusoidal positional embedding matrix."""
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # shape: (max_seq_len, embedding_dim)
    
    def forward(self, x, context):
        """
        Args:
          x: LongTensor of shape (batch_size, seq_len) – the user behavior sequence (item IDs).
          context: List of LongTensors, each of shape (batch_size,) – additional context features.
        Returns:
          logits: Tensor of shape (batch_size, num_items), the unnormalized scores.
          targets: Tensor of shape (batch_size,) assumed to be the target (last item in sequence).
        """
        batch_size, seq_len = x.size()
        # Get item embeddings and scale by sqrt(embedding_dim)
        item_emb = self.item_embed(x) * math.sqrt(self.config['item_embed']['embedding_dim'])
        # Add positional encoding
        if self.config.get('use_sinusoidal_pos', True):
            pos_emb = self.pos_embedding_buffer[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            item_emb = item_emb + pos_emb
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embed(positions)
            item_emb = item_emb + pos_emb
        
        # Create mask (assumes padding index is 0)
        mask = (x != self.item_embed.padding_idx)  # shape: (batch_size, seq_len)
        
        # Pass through transformer encoder
        enc_out = self.encoder(item_emb, mask=mask)
        # Aggregate encoder outputs (mean pooling over valid positions)
        mask_unsq = mask.unsqueeze(-1).float()
        sum_enc = torch.sum(enc_out * mask_unsq, dim=1)
        len_enc = torch.clamp(mask_unsq.sum(dim=1), min=1e-9)
        agg_encoding = sum_enc / len_enc  # shape: (batch_size, d_model)
        
        # Process context features: embed each and concatenate
        context_embs = []
        for emb, feat in zip(self.context_embeddings, context):
            feat_tensor = feat.long().to(x.device)
            context_embs.append(emb(feat_tensor))
        context_cat = torch.cat(context_embs, dim=1) if context_embs else torch.empty(batch_size, 0, device=x.device)
        
        # Concatenate aggregated transformer output with context embeddings
        mlp_input = torch.cat([agg_encoding, context_cat], dim=1)
        logits = self.mlp(mlp_input)
        # Here we assume that the target item is the last item in the sequence
        targets = x[:, -1]
        return logits, targets

#################################
# Trainer and Data Preparation  #
#################################

class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.grad_clipping = config.get('grad_clipping', False)
        if self.grad_clipping:
            self.clipper = GradientClipping(config['clip_value'])
            # Register hook on all modules that have a weight parameter
            self.model.apply(lambda module: self.clipper.register_hook(module) if hasattr(module, 'weight') else None)
        self.scheduler = None
    
    def set_lr_scheduler(self, milestones, gamma, last_epoch=-1):
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        for x_batch, context_batch in train_loader:
            x_batch = x_batch.to(self.device)
            context_batch = [c.to(self.device) for c in context_batch]
            logits, targets = self.model(x_batch, context_batch)
            loss = self.loss_fn(logits, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipper.clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        elapsed = time.time() - start_time
        print(f"Training loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
        return avg_loss

def pad_sequence(seq, max_seq_len, pad_value=0):
    """Pads (or truncates) a sequence to a fixed maximum length."""
    seq = list(seq)
    if len(seq) >= max_seq_len:
        return seq[-max_seq_len:]
    else:
        return [pad_value] * (max_seq_len - len(seq)) + seq

def batch_fn(user_sequences, context_features, batch_size, max_seq_len, shuffle=True):
    """
    Prepares batches from user behavior sequences and corresponding context features.
    Args:
      user_sequences: list of lists (each inner list is a sequence of item IDs)
      context_features: list of lists, each corresponding to one additional feature
      batch_size: number of samples per batch
      max_seq_len: fixed sequence length for the transformer input
      shuffle: whether to shuffle the data before batching
    Yields:
      A tuple (x_batch, context_batch) where:
        - x_batch is a LongTensor of shape (batch_size, max_seq_len)
        - context_batch is a list of LongTensors (one per feature) of shape (batch_size,)
    """
    data = list(zip(user_sequences, *context_features))
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        if len(batch) < batch_size:
            continue  # skip incomplete batch for simplicity
        seqs = [pad_sequence(item[0], max_seq_len) for item in batch]
        # For each context feature (starting from index 1)
        context_batches = []
        num_context = len(batch[0]) - 1
        for j in range(num_context):
            feat_batch = [item[j+1] for item in batch]
            context_batches.append(torch.LongTensor(feat_batch))
        yield torch.LongTensor(seqs), context_batches

##############################
# Example Usage (Main Loop)  #
##############################

if __name__ == "__main__":
    # Dummy configuration (adjust these numbers as needed)
    config = {
        'item_embed': {
            'num_embeddings': 10000,  # total number of items in the catalog
            'embedding_dim': 64,
            'padding_idx': 0
        },
        'context_features': [
            # Example: user age (vocabulary size 100, embedding dimension 8)
            {'num_embeddings': 100, 'embedding_dim': 8, 'padding_idx': 0},
            # Example: user gender (vocabulary size 3, embedding dimension 4)
            {'num_embeddings': 3, 'embedding_dim': 4, 'padding_idx': 0}
        ],
        'max_seq_len': 20,
        'trans': {
            'input_size': 64,    # should match item_embed.embedding_dim
            'hidden_size': 128,
            'n_layers': 1,
            'n_heads': 8
        },
        'lr': 0.001,
        'clip_value': 1.0,
        'grad_clipping': True,
        'init_method': 'xavier',
        'use_sinusoidal_pos': True,
        'dropout': 0.1
    }
    
    # Instantiate model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSTransformer(config)
    trainer = Trainer(model, config, device)
    
    # Generate dummy data (for demonstration purposes)
    num_samples = 1000
    user_sequences = []
    context_age = []
    context_gender = []
    for _ in range(num_samples):
        seq_length = random.randint(5, config['max_seq_len'])
        # Generate a sequence of item IDs (non-zero values)
        seq = [random.randint(1, config['item_embed']['num_embeddings'] - 1) for _ in range(seq_length)]
        user_sequences.append(seq)
        context_age.append(random.randint(1, 99))
        context_gender.append(random.randint(0, 2))
    
    # Create data loader (a generator of batches)
    batch_size = 32
    train_loader = list(batch_fn(user_sequences, [context_age, context_gender], batch_size, config['max_seq_len']))
    
    # Train for one epoch
    trainer.train_epoch(train_loader)
