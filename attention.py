import math
import torch
class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__() #thread the constructors together
        # three matrices which give a different view of the same token
        self.embed_dim = embed_dim
        # create a matrice [embed_dim x embed_dim]
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, token_embed, attention_mask):
        # seq_len is the number of rows
        seq_len = token_embed.shape[0]

        # then create matrices
        Q = self.W_q(token_embed)
        K = self.W_k(token_embed)
        V = self.W_v(token_embed)

        # calculate attention score
        scores = (Q @ K.T) / math.sqrt(self.embed_dim)

        # casual mask
        causal = torch.triu(torch.ones(seq_len, seq_len, device=token_embed.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))

        # padding mask
        if attention_mask is not None:
            key_mask = (attention_mask == 0).unsqueeze(0)
            scores = scores.masked_fill(key_mask, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        return out, attn