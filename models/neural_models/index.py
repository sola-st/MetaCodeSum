class Index():
	def __init__(self):
		self.word2index = {"PAD":0,"OOV":1}
		self.index2word = {0:"PAD",1:"OOV"}
		self.wordID = 2
		print("initializing index")
		

	def add_word_from_token_list(self,tokens):
		for token in tokens:
			self.__add_word_to_index(token)

	def add_word_from_embedding(self,embedding):
		for word in embedding.wv.vocab:			
			self.__add_word_to_index(word)
		print("{0},{1}".format(len(self.word2index),len(self.index2word)))

	def add_initialized_index(self,word2index,index2word):
		self.word2index = word2index
		self.index2word = index2word
		self.wordID = len(self.word2index)


	def __add_word_to_index(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.wordID
			self.index2word[self.wordID] = word
			self.wordID += 1

	def vocab_size(self):
		return len(self.index2word)

	