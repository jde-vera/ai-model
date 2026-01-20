import torch
from transformer import Transformer
from attention import SelfAttention
from block import TransformerBlock

if __name__ == '__main__':
    transformer = Transformer()
    usr_input = "Miley Cyrus - Party In The U.S.A"
    transformer.build_vocab(usr_input)
    ids = transformer.get_ids()
    print(ids) # this is to check if the new local variable imp. works
    id_pad, attention_mask = transformer.pad_and_mask(ids, 5)

    # you need to assign the self.embedding variable with an embedding object in order to embed token ids
    transformer.build_embeddings(16)
    token_vecs = transformer.embed(id_pad) # create dense vectors for tokens
    transformer.build_positional_embeddings(max_len=5, embed_dim=16)
    token_embed = transformer.add_positions(token_vecs) # create embeddings for positions and add to token vectors
    
    # ===transformer block===
    embed_dim = 16
    hidden_dim = 4 * embed_dim
    block = TransformerBlock(embed_dim, hidden_dim)

    mask = torch.tensor(attention_mask, dtype=torch.long)
    out, attn = block(token_embed, attention_mask=mask)

    print("token_embed shape:", token_embed.shape)
    print("out shape:", out.shape)
    print("attn shape:", attn.shape)