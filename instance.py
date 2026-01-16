from transformer import Transformer

if __name__ == '__main__':
    transformer = Transformer()
    usr_input = "Miley Cyrus - Party In The U.S.A"
    transformer.build_vocab(usr_input)
    ids = transformer.get_ids()
    transformer.pad_and_mask(ids, 5)

    transformer.build_embeddings(16)
    print(transformer.embed(ids))