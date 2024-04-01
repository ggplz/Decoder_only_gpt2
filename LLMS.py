import math
import os

import requests
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
context_length = 16  # 16 个 token
d_model = 64
num_blocks = 8
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = 'mps'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)
with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# tokenize the text
encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text)

# split into train and validation
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
value_data = tokenized_text[train_size:]


class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# define Scaled Dot product attention
class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.context_length, self.context_length)))
        self.dropout_layer = nn.Dropout(self.dropout)
        # self.wq = nn.Linear(d_model, d_model)
        # self.wk = nn.Linear(d_model, d_model)
        # self.wv = nn.Linear(d_model, d_model)
        # apply mask

        # self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)
        # Q = x @ self.wq
        # K = x @ self.wk
        # V = x @ self.wv
        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention_scores = attention_scores.masked_fill((self.tril[:T, :T]) == 0, float('-inf'))
        attention_scores = F.softmax(input=attention_scores, dim=-1)
        attention_scores = self.dropout_layer(attention_scores)
        out = attention_scores @ v

        # attention = Q @ K.transpose(-2, -1) / math.sqrt(d_model // num_heads)
        # attention = attention.masked_fill(self.mask == 0, float('-inf'))
        # attention = F.softmax(attention, -1)
        # out = attention @ V

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = Feedforward()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        x = x + self.multi_head_attention_layer(self.layer_norm1(x))
        x = x + self.feed_forward_layer(self.layer_norm2(x))

        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1,
                                                         embedding_dim=self.d_model)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] + [
            nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)

        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_rashaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_rashaped)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length:]
            logits, loss = self(idx_crop)
            logits_last_timestep = logits[:, -1, :]
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = TransformerLanguageModel()
model = model.to(device)


def get_batch(split: str):
    data = train_data if split == 'train' else value_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([torch.tensor(data[idx:idx + context_length], dtype=torch.long) for idx in idxs]).to(device)
    y = torch.stack([torch.tensor(data[idx + 1:idx + context_length + 1], dtype=torch.long) for idx in idxs]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('Train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pt')

model.eval()
start = 'The ggplz is a'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('``````````````````')
print(encoding.decode(y[0].tolist()))
print('``````````````````')
