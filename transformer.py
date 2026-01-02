import re

class Transformer:
    
    
    def tokenize(self, user_input):
        '''
        the tokenize() take user input (string) and makes use of re
        '''
        
        regex = re.compile(
            r"(?:[A-Za-z]+(?:[-'][A-Za-z]+)*) | (?:\d+(?:\.\d+)?) | (?:[^\w\s])", re.VERBOSE)

        return regex.findall(user_input)