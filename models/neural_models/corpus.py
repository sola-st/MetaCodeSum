import torch
import codecs
import util as ut
import numpy as np
import os
import json
from torch.autograd import Variable
from code_tokenizer import tokenize

class Dataset(torch.utils.data.Dataset):
	def __init__(self, file_name,code_index,nl_index):
		self.source_file = file_name		
		self.code_index = code_index
		self.nl_index = nl_index
		self.data_map = {}
		self.__read_source_file()
		
		self.code = "code"
		self.body_tensor = "body_tensor"

		self.summary = "summary"
		self.summary_tensor = "summary_dict"
		
		self.label = "label"
		self.label_tensor = "label_tensor"
		return

	def __len__(self):
		return len(self.data_map)

	def __read_source_file(self):		
		with open(self.source_file, "r") as f:
			data_list = json.load(f)
			for count,elms in enumerate(data_list):
				if count not in self.data_map:
					self.data_map[count] = elms

		print("read {} items in data_map".format(len(self.data_map)))

	def __getitem__(self,idx):
		if idx in self.data_map:
			elms = self.data_map[idx]
			
			if self.body_tensor not in elms:
				elms[self.body_tensor] = self.__tensorFromBody(elms[self.code])

			if self.summary_tensor not in elms:
				elms[self.summary_tensor] = self.__tensorForSummaries(elms[self.summary])

			if self.label_tensor not in elms:
				elms[self.label_tensor] = self.__tensorForLabel(elms[self.label])
			
			# update the element with body tensor and label tensor
			self.data_map[idx] = elms
			m1_label,m2_label,m3_label = elms[self.label_tensor]
			m1_tensor,m2_tensor,m3_tensor = elms[self.summary_tensor]
			# return values
			return {'body': elms[self.body_tensor],ut.model_names[0]: m1_tensor,ut.model_names[1]:m2_tensor,ut.model_names[2]:m3_tensor,
			'label1': m1_label,'label2': m2_label,'label3': m3_label,"_id":elms["_id"]}
			

	def __tensorFromBody(self, method_body):
		
		indexes =[]
		for word in tokenize(method_body):
			if word in self.code_index.word2index:
				indexes.append(self.code_index.word2index[word])
			else:
				indexes.append(1) # OOV term

		pad_input =[]
		if len(indexes)> ut.seq_len:
			pad_input=indexes[0:ut.seq_len]
		else:
			dif = ut.seq_len - len(indexes)
			for i in range(dif):
				indexes.append(0)
			pad_input = indexes
		
		return torch.tensor(pad_input, dtype=torch.long)

	def __tensorForSummaries(self,summary_arr):
		m1_tensor = self.__tensorFromNLTokens(summary_arr[ut.model_names[0]])
		m2_tensor = self.__tensorFromNLTokens(summary_arr[ut.model_names[1]])
		m3_tensor = self.__tensorFromNLTokens(summary_arr[ut.model_names[2]])

		return m1_tensor,m2_tensor,m3_tensor

	def __tensorFromNLTokens(self, tokens):
		
		indexes =[]
		for word in tokenize(tokens):
			if word in self.nl_index.word2index:
				indexes.append(self.nl_index.word2index[word])
			else:
				indexes.append(1) # OOV term

		pad_input =[]
		if len(indexes)> ut.nl_seq_len:
			pad_input=indexes[0:ut.nl_seq_len]
		else:
			dif = ut.nl_seq_len - len(indexes)
			for i in range(dif):
				indexes.append(0)
			pad_input = indexes
		
		return torch.tensor(pad_input, dtype=torch.long)


	def __tensorForLabel(self, raw_labels):		
		m1_label = torch.tensor(self.__add_label_stat(ut.model_names[0],raw_labels), dtype=torch.float32)
		m2_label = torch.tensor(self.__add_label_stat(ut.model_names[1],raw_labels), dtype=torch.float32)
		m3_label = torch.tensor(self.__add_label_stat(ut.model_names[2],raw_labels), dtype=torch.float32)

		return m1_label,m2_label,m3_label

	def __add_label_stat(self,model_name,raw_labels):
		if raw_labels[model_name] == True:
			return [1.0]
		else:
			return [0.0]
		
	