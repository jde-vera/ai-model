import torch
from attention import SelfAttention

class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.ln1 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_out, attn = self.attn(x, attention_mask=attention_mask)
        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        return x, attn
