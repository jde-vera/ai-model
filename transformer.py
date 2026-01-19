import re
import torch

class Transformer:
    def __init__(self):
        self.vocab = {}
        self.embedding = []
        self.pos_embedding = []

    def tokenize(self, user_input):
        '''
        the tokenize() take user input (string) and makes use of re
        '''
        
        regex = re.compile(
            r"(?:[A-Za-z]+(?:[-'][A-Za-z]+)*) | (?:\d+(?:\.\d+)?) | (?:[^\w\s])", re.VERBOSE)

        return regex.findall(user_input)
    
    def build_vocab(self, user_input):
        '''
        Docstring for build_vocab
        
        :param self: Description
        '''

        token_list = self.tokenize(user_input)

        tokens = set()

        for token in token_list:
            tokens.add(token)
        
        self.vocab = {"[PAD]": 0, "[UNK]": 1}

        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def get_ids(self):
        '''
        Docstring for get_ids
        
        :param self: Description
        this returns a list of ids from a given sequence, which will be used in the pad_and_mask() 
        '''
        id_list = []
        for id in self.vocab.values():
            id_list.append(id)
        return id_list
    def pad_and_mask(self, id, max_len):
        '''
        Docstring for pad_and_mask
        
        :param self: Description
        :param id: list of token ids
        :param max_len: max len of id and attention_mask lists
        this truncates a given list of ids if the list is too big
        and pads if the list is too small
        0 indicates padding and 1 indicates real token
        '''
        id = id[:max_len]

        attention_mask = [1] * len(id)

        pad = max_len - len(id)
        if pad > 0:
            id = id + ([0] * pad)
            attention_mask = attention_mask + ([0] * pad)

        return id, attention_mask
    def build_embeddings(self, embed_dim):
        '''
        Docstring for build_embeddings
        
        :param self: Description
        create a word embeddings table for the vocabulary
        '''
        vocab_size = len(self.vocab)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim)
    
    def embed(self, input_ids):
        '''
        Docstring for embed
        
        :param self: Description
        :param input_ids: list of token ids
        converts the token ids into embedding vectors
        '''
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return self.embedding(ids_tensor)
    def add_positions(self, token_vecs):
        '''
        Docstring for add_positions
        
        :param self: Description
        :param token_vecs: tensor object
        add positions so that the transformer knows the order
        '''
        seq_len = token_vecs.shape[0]
        positions = torch.arange(seq_len, dtype=torch.long)
        pos_vecs = self.pos_embedding(positions)
        return token_vecs + pos_vecs
    def build_positional_embeddings(self, max_len, embed_dim):
        '''
        Docstring for build_positional_embeddings
        
        :param self: Description
        :param max_len: the number of tokens the model is allow to see at once
        :param embed_dim: how many digits you want to represent the dense vectors
        '''
        self.pos_embedding = torch.nn.Embedding(max_len, embed_dim)