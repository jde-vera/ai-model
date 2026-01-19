from transformer import Transformer

if __name__ == '__main__':
    transformer = Transformer()
    usr_input = "Miley Cyrus - Party In The U.S.A"
    transformer.build_vocab(usr_input)
    ids = transformer.get_ids()
    transformer.pad_and_mask(ids, 5)

    # you need to assign the self.embedding variable with an embedding object in order to embed token ids
    transformer.build_embeddings(16)
    print(transformer.embed(ids)) 

    print(type(transformer.embed(ids)))