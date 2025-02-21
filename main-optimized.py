import math
import random
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# Positional Encoding (Sinusoidal)  

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    Given an input of shape (batch_size, seq_len, d_model),
    returns the input plus positional encodings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Multi-head Self-Attention Module 

class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention.
    This version optionally uses flash attention via torch.nn.functional.scaled_dot_product_attention.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash_attention=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        # Linear projections
        q = self.q_linear(x)  # (B, seq_len, d_model)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # Reshape into (B, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if self.use_flash_attention:
            # For flash attention, create an attn_mask of shape (B, seq_len, seq_len)
            # Here, our original mask is (B, seq_len) with True for valid tokens.
            if mask is not None:
                # In flash attention, True indicates positions that should be masked out.
                attn_mask = ~mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            else:
                attn_mask = None
            # F.scaled_dot_product_attention accepts inputs of shape (B, n_heads, L, d)
            out = F.scaled_dot_product_attention(q, k, v,
                                                 attn_mask=attn_mask,
                                                 dropout_p=self.dropout.p,
                                                 is_causal=False)
        else:
            # Standard scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                # Expand mask: (B, seq_len) -> (B, 1, 1, seq_len)
                scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        # Concatenate heads: reshape from (B, n_heads, seq_len, d_k) to (B, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.fc(out)
        return out

# Feed-Forward Network Module       

class FeedForward(nn.Module):
    """
    Point-wise feed-forward network.
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# Transformer Block (Encoder Cell)  

class TransformerBlock(nn.Module):
    """
    A single transformer encoder block consisting of:
      - Multi-head self-attention with residual connection and layer norm.
      - Feed-forward network with residual connection and layer norm.
    """
    def __init__(self, d_model, n_heads, ff_hidden_dim, dropout=0.1, use_flash_attention=False):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, use_flash_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # In-place addition to save memory
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# Transformer Encoder              

class TransformerEncoder(nn.Module):
    """
    Stacks several transformer blocks.
    """
    def __init__(self, d_model, n_heads, ff_hidden_dim, n_layers, dropout=0.1, use_flash_attention=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden_dim, dropout, use_flash_attention)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Behavior Sequence Transformer     

class BSTransformer(nn.Module):
    """
    Behavior Sequence Transformer (BST) for CTR prediction.
    - Embeds a user behavior sequence of item IDs.
    - Adds positional encoding.
    - Uses a transformer encoder to capture sequential dependencies.
    - Extracts the target item representation (assumed to be the last token).
    - Concatenates it with embeddings of other context features.
    - Passes the result through a three-layer MLP to output logits.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        
        # Embedding for sequence items.
        self.item_embedding = nn.Embedding(
            num_embeddings=config['item_vocab_size'],
            embedding_dim=d_model,
            padding_idx=config['padding_idx']
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=config['max_seq_len'])
        # Transformer encoder (stacked transformer blocks)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=config['n_heads'],
            ff_hidden_dim=config['ff_hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config.get('dropout', 0.1),
            use_flash_attention=config.get('use_flash_attention', False)
        )
        # Context feature embeddings (e.g., user profile features)
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=feat['vocab_size'],
                embedding_dim=feat['embed_dim'],
                padding_idx=feat.get('padding_idx', 0)
            ) for feat in config['context_features']
        ])
        # MLP: Concatenates transformer output (target item representation)
        # with all context feature embeddings.
        mlp_input_dim = d_model + sum(feat['embed_dim'] for feat in config['context_features'])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, config['mlp_hidden1']),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config['mlp_hidden1'], config['mlp_hidden2']),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config['mlp_hidden2'], config['item_vocab_size'])
        )
        self._init_weights()
        
    def _init_weights(self):
        init_method = self.config.get('init_method', 'xavier')
        for param in self.parameters():
            if param.dim() > 1:
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(param)
    
    def forward(self, x, context_list):
        """
        Args:
          x: LongTensor of shape (B, seq_len) with item IDs.
          context_list: List of LongTensors (each of shape (B,)) for additional features.
        Returns:
          logits: Tensor of shape (B, item_vocab_size) with unnormalized scores.
          targets: Tensor of shape (B,) representing the ground-truth target item (last token).
        """
        batch_size, seq_len = x.size()
        # Embed items and scale by sqrt(d_model)
        emb = self.item_embedding(x) * math.sqrt(self.config['d_model'])
        # Add positional encoding
        emb = self.pos_encoder(emb)
        # Create mask: valid tokens are those not equal to padding_idx
        mask = (x != self.config['padding_idx'])
        # Pass through transformer encoder
        transformer_out = self.transformer(emb, mask)
        # Use representation of the last token as target representation
        target_repr = transformer_out[:, -1, :]  # (B, d_model)
        # Embed each context feature and concatenate
        context_repr = [emb_layer(ctx.to(x.device)) for emb_layer, ctx in zip(self.context_embeddings, context_list)]
        if context_repr:
            context_concat = torch.cat(context_repr, dim=1)
        else:
            context_concat = torch.empty(batch_size, 0, device=x.device)
        # Concatenate and pass through MLP
        mlp_input = torch.cat([target_repr, context_concat], dim=1)
        logits = self.mlp(mlp_input)
        targets = x[:, -1]
        return logits, targets

# Data Batching Helper Functions    

def pad_seq(seq, max_len, pad_value=0):
    """
    Pads (or truncates) a sequence to a fixed maximum length.
    Pads on the left so that the target (last token) is preserved.
    """
    seq = list(seq)
    if len(seq) < max_len:
        return [pad_value] * (max_len - len(seq)) + seq
    else:
        return seq[-max_len:]

def batch_generator(user_sequences, context_features, batch_size, max_seq_len):
    """
    Yields batches of data.
      - user_sequences: list of lists (each inner list is a sequence of item IDs)
      - context_features: list of lists, one per additional feature
    """
    data = list(zip(user_sequences, *context_features))
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if len(batch) < batch_size:
            continue  # Skip incomplete batch for simplicity
        seqs = [pad_seq(item[0], max_seq_len) for item in batch]
        context_batches = []
        num_ctx = len(batch[0]) - 1
        for j in range(num_ctx):
            ctx_batch = [item[j + 1] for item in batch]
            context_batches.append(torch.LongTensor(ctx_batch))
        yield torch.LongTensor(seqs), context_batches

# Trainer (with Optional AMP)      

class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = None
        # Optionally use automatic mixed precision
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
    def set_lr_scheduler(self, milestones, gamma, last_epoch=-1):
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        start_time = time.time()
        for x_batch, ctx_batch in train_loader:
            x_batch = x_batch.to(self.device)
            ctx_batch = [ctx.to(self.device) for ctx in ctx_batch]
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, targets = self.model(x_batch, ctx_batch)
                    loss = self.loss_fn(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, targets = self.model(x_batch, ctx_batch)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item()
            batch_count += 1
        print(f"Epoch Loss: {total_loss / batch_count:.4f} | Time: {time.time() - start_time:.2f}s")

# main loop        

if __name__ == "__main__":
    # Configuration parameters inspired by the paper.
    config = {
        'item_vocab_size': 10000,     # total number of items
        'd_model': 64,                # transformer and embedding dimension
        'padding_idx': 0,
        'max_seq_len': 20,
        'n_heads': 8,
        'ff_hidden_dim': 128,
        'n_layers': 1,                # b = 1 as found optimal
        'dropout': 0.1,
        'use_flash_attention': True,  # switch on flash attention optimizations
        'use_amp': True,              # optional: use mixed precision
        'context_features': [         # example context features
            {'vocab_size': 100, 'embed_dim': 8},
            {'vocab_size': 3, 'embed_dim': 4}
        ],
        'mlp_hidden1': 1024,
        'mlp_hidden2': 512,
        'lr': 0.001,
        'init_method': 'xavier'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSTransformer(config)
    trainer = Trainer(model, config, device)

    num_samples = 1000
    user_sequences = []
    context_feature1 = []  # e.g., user age
    context_feature2 = []  # e.g., user gender
    for _ in range(num_samples):
        seq_len = random.randint(5, config['max_seq_len'])
        seq = [random.randint(1, config['item_vocab_size'] - 1) for _ in range(seq_len)]
        user_sequences.append(seq)
        context_feature1.append(random.randint(1, 99))
        context_feature2.append(random.randint(0, 2))
    
    batch_size = 32
    train_loader = list(batch_generator(user_sequences, [context_feature1, context_feature2],
                                        batch_size, config['max_seq_len']))
    

    trainer.train_epoch(train_loader)
