## pipeline
1. essentially the user passes a given string -> that string is then turned into a sequence of tokens
2. build a dictionary where the keys are the tokens and assign a token id (in my model it is the current len(self.vocab))
3. create id list and pad or truncate -> create attention mask and do the same 
4. create an embedding object -> user specifies the dimensions (how many digits used to represent the token) and the no. of rows is the number of tokens
5. then create the matrix using the embedding object -> get the ids of the tokens and then turn it into a tensor object so the embedding object can use it