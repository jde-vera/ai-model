import re

class Transformer:
    
    '''
    the tokenize() take user input (string) and makes use of re
    '''
    def tokenize(self, user_input):
        regex = re.compile(
            r"(?:[A-Za-z]+(?:[-'][A-Za-z]+)*) | (?:\d+(?:\.\d+)?) | (?:[^\w\s])", re.VERBOSE)

        return regex.findall(user_input)