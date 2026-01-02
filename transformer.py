import re

class Transformer:
    def __init__(self):
        self.vocab = {}

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