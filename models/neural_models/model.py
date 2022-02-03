import torch
import torch.nn as nn
import torch.nn.functional as F
import util as ut
import math

class LSTM_Meta(nn.Module):
	"""
	Implementation of LSTM based meta-model.
	"""
	def __init__(self, code_embeddings,nl_embeddings, code_vocab_size,nl_vocab_size, tagset_size):
		super(LSTM_Meta, self).__init__()
		
		# init code embedding layer
		self.code_embeddings = None
		if code_embeddings == None:
			self.code_embeddings = nn.Embedding(code_vocab_size, ut.embedding_dim)
		else:
			self.code_embeddings = nn.Embedding.from_pretrained(code_embeddings,freeze= False)

		#init nl embedding layer
		self.nl_embeddings = None
		if nl_embeddings == None:
			self.nl_embeddings = nn.Embedding(nl_vocab_size, ut.embedding_dim)
		else:
			self.nl_embeddings = nn.Embedding.from_pretrained(nl_embeddings,freeze= False)

		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
		self.pl_token_lstm 		= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.code_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.code_model_lstm 	= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.seq_model_lstm 	= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.trn_model_lstm 	= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)

		# joint layers, that takes input code lstm out and model lstm
		self.joint_code = nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with code model out
		self.joint_seq 	= nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with seq model out
		self.joint_trn 	= nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with trn model out

		self.joint_drop_out = nn.Dropout(0.15)
		# linear layer on joint layers	
		self.fc_code = nn.Linear(ut.joint_out, ut.hidden_out)
		self.fc_seq = nn.Linear(ut.joint_out, ut.hidden_out)
		self.fc_trn = nn.Linear(ut.joint_out, ut.hidden_out)

		self.fc_drop_out = nn.Dropout(0.15)
		# The linear layer that maps scaled representation to tag space
		self.fc_m1 = nn.Linear(ut.hidden_out, 1)
		self.fc_m2 = nn.Linear(ut.hidden_out, 1)
		self.fc_m3 = nn.Linear(ut.hidden_out, 1)
		

	def forward(self, input_dict):
		# extract individual values
		pl_tokens 	= input_dict["code_tokens"]
		code_model 	= input_dict[ut.model_names[0]]
		seq_model 	= input_dict[ut.model_names[1]]
		trn_model 	= input_dict[ut.model_names[2]]

		
		#embedding representations
		pl_embeds 			= self.code_embeddings(pl_tokens)
		code_model_embeds 	= self.nl_embeddings(code_model)
		seq_embeds			= self.nl_embeddings(seq_model)
		trn_embeds 			= self.nl_embeddings(trn_model)

		
		# embedded output = [batch_first size, sent_len, emb dim]
		pl_1_lstm_out,(pl_1_hn,_) 	= self.pl_token_lstm(pl_embeds)		
		m1_lstm_out,(code_hn,_) 	= self.code_model_lstm(code_model_embeds)
		m2_lstm_out,(seq_hn,_) 		= self.seq_model_lstm(seq_embeds)
		m3_lstm_out,(trn_hn,_) 		= self.trn_model_lstm(trn_embeds)
		

		#concat the final forward and backward hidden state, of 
		pl_1_hidden 	= F.relu(torch.cat((pl_1_hn[0,:,:], pl_1_hn[-1,:,:]), dim = 1))
		code_hidden 	= F.relu(torch.cat((code_hn[0,:,:], code_hn[-1,:,:]), dim = 1))
		seq_hidden 		= F.relu(torch.cat((seq_hn[0,:,:], seq_hn[-1,:,:]), dim = 1))
		trn_hidden 		= F.relu(torch.cat((trn_hn[0,:,:], trn_hn[-1,:,:]), dim = 1))

		# concat pl with nl tokens 
		joint_code_in 	= F.relu(torch.cat((pl_1_hidden, code_hidden), dim = 1))
		joint_seq_in 	= F.relu(torch.cat((pl_1_hidden, seq_hidden), dim = 1))
		joint_trn_in 	= F.relu(torch.cat((pl_1_hidden, trn_hidden), dim = 1))
		
		# joint layer
		joint_code_out = self.joint_drop_out(self.joint_code(joint_code_in))
		joint_seq_out = self.joint_drop_out(self.joint_seq(joint_seq_in))
		joint_trn_out = self.joint_drop_out(self.joint_trn(joint_trn_in))
		
		#linear layers on joint out
		fc_code_out = self.fc_drop_out(F.relu(self.fc_code(joint_code_out)))
		fc_seq_out = self.fc_drop_out(F.relu(self.fc_seq(joint_seq_out)))
		fc_trn_out = self.fc_drop_out(F.relu(self.fc_trn(joint_trn_out)))
		
		# final layers
		out_1 = torch.sigmoid(self.fc_m1(fc_code_out))
		out_2 = torch.sigmoid(self.fc_m2(fc_seq_out))
		out_3 = torch.sigmoid(self.fc_m3(fc_trn_out))
		
		return  out_1,out_2,out_3


class TRN_Meta(nn.Module):
	"""
	Implementation of Transformer based meta-model.
	"""
	def __init__(self, code_embeddings,nl_embeddings, code_vocab_size,nl_vocab_size, tagset_size):
		super(TRN_Meta, self).__init__()
		
		# init code embedding layer
		self.code_embeddings = None
		if code_embeddings == None:
			self.code_embeddings = nn.Embedding(code_vocab_size, ut.embedding_dim)
		else:
			self.code_embeddings = nn.Embedding.from_pretrained(code_embeddings,freeze= False)

		#init nl embedding layer
		self.nl_embeddings = None
		if nl_embeddings == None:
			self.nl_embeddings = nn.Embedding(nl_vocab_size, ut.embedding_dim)
		else:
			self.nl_embeddings = nn.Embedding.from_pretrained(nl_embeddings,freeze= False)

		# declare positional encoding for nl and pl tokens
		self.code_1_pos_encoder = PositionalEncoding(d_model=ut.embedding_dim,vocab_size=code_vocab_size)
		self.nl_1_pos_encoder 	= PositionalEncoding(d_model=ut.embedding_dim,vocab_size=nl_vocab_size)
		self.nl_2_pos_encoder 	= PositionalEncoding(d_model=ut.embedding_dim,vocab_size=nl_vocab_size)
		self.nl_3_pos_encoder 	= PositionalEncoding(d_model=ut.embedding_dim,vocab_size=nl_vocab_size)

		# encode token sequence with transformer

		self.code_1_encoder_layer 	= nn.TransformerEncoderLayer(d_model=ut.embedding_dim, nhead=ut.head_count, dim_feedforward=ut.ff_count)
		self.code_1_encoder 		= nn.TransformerEncoder(self.code_1_encoder_layer, num_layers=ut.encoder_layer)
		
		self.m1_encoder_layer 	= nn.TransformerEncoderLayer(d_model=ut.embedding_dim, nhead=ut.head_count, dim_feedforward=ut.ff_count)
		self.m1_encoder 		= nn.TransformerEncoder(self.m1_encoder_layer, num_layers=ut.encoder_layer)

		self.m2_encoder_layer 	= nn.TransformerEncoderLayer(d_model=ut.embedding_dim, nhead=ut.head_count, dim_feedforward=ut.ff_count)
		self.m2_encoder 		= nn.TransformerEncoder(self.m2_encoder_layer, num_layers=ut.encoder_layer)

		self.m3_encoder_layer 	= nn.TransformerEncoderLayer(d_model=ut.embedding_dim, nhead=ut.head_count, dim_feedforward=ut.ff_count)
		self.m3_encoder 		= nn.TransformerEncoder(self.m3_encoder_layer, num_layers=ut.encoder_layer)
		
		# lstm layers
		self.pl1_token_lstm 	= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.code_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.m1_model_lstm 		= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.m2_model_lstm 		= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)
		self.m3_model_lstm 		= nn.LSTM(input_size=ut.embedding_dim,hidden_size=ut.nl_lstm_out_dim, batch_first=True,num_layers=2,bidirectional=True)

		# joint layers, that takes input code lstm out and model lstm
		self.joint_m1 			= nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with code model out
		self.joint_m2 			= nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with seq model out
		self.joint_m3 			= nn.Linear(ut.joint_in_size, ut.joint_out) # pl rep joined with trn model out

		self.joint_drop_out 	= nn.Dropout(0.15)
		# linear layer on joint layers	
		self.fc_m1 			= nn.Linear(ut.joint_out, ut.hidden_out)
		self.fc_m2 			= nn.Linear(ut.joint_out, ut.hidden_out)
		self.fc_m3 			= nn.Linear(ut.joint_out, ut.hidden_out)

		self.fc_drop_out 		= nn.Dropout(0.15)
		# The linear layer that maps scaled representation to tag space
		self.fc_pred_m1 		= nn.Linear(ut.hidden_out, 1)
		self.fc_pred_m2 		= nn.Linear(ut.hidden_out, 1)
		self.fc_pred_m3 		= nn.Linear(ut.hidden_out, 1)

	def forward(self,input_dict):
		# extract individual values
		pl_tokens 	= input_dict["code_tokens"]
		m1_model 	= input_dict[ut.model_names[0]]
		m2_model 	= input_dict[ut.model_names[1]]
		m3_model 	= input_dict[ut.model_names[2]]
		# get embeddings
		pl_1_embeds = self.code_embeddings(pl_tokens)
		m1_embeds 	= self.nl_embeddings(m1_model)
		m2_embeds	= self.nl_embeddings(m2_model)
		m3_embeds 	= self.nl_embeddings(m3_model)

		# merge with positional encodings
		pl_1_pos_enc = self.code_1_pos_encoder(pl_1_embeds)
		m1_pos_enc	= self.nl_1_pos_encoder(m1_embeds) 
		m2_pos_enc	= self.nl_2_pos_encoder(m2_embeds) 
		m3_pos_enc	= self.nl_3_pos_encoder(m3_embeds)
		
		# trn encoded	
		pl_1_trn_enc  	= self.code_1_encoder(pl_1_pos_enc)
		m1_trn_enc  	= self.m1_encoder(m1_pos_enc)
		m2_trn_enc  	= self.m2_encoder(m2_pos_enc)
		m3_trn_enc  	= self.m3_encoder(m3_pos_enc)
		# lstm out
		pl_1_lstm_out,(pl_1_hn,_) 	= self.pl1_token_lstm(pl_1_trn_enc)
		m1_lstm_out,(m1_hn,_) 		= self.m1_model_lstm(m1_trn_enc)
		m2_lstm_out,(m2_hn,_) 		= self.m2_model_lstm(m2_trn_enc)
		m3_lstm_out,(m3_hn,_) 		= self.m3_model_lstm(m3_trn_enc)

		#concat the final forward and backward hidden state, of 
		pl_1_hidden = F.relu(torch.cat((pl_1_hn[0,:,:], pl_1_hn[-1,:,:]), dim = 1))
		m1_hidden 	= F.relu(torch.cat((m1_hn[0,:,:], m1_hn[-1,:,:]), dim = 1))
		m2_hidden 	= F.relu(torch.cat((m2_hn[0,:,:], m2_hn[-1,:,:]), dim = 1))
		m3_hidden 	= F.relu(torch.cat((m3_hn[0,:,:], m3_hn[-1,:,:]), dim = 1))
		
		# concat pl with nl tokens 
		joint_m1_in 	= F.relu(torch.cat((pl_1_hidden, m1_hidden), dim = 1))
		joint_m2_in 	= F.relu(torch.cat((pl_1_hidden, m2_hidden), dim = 1))
		joint_m3_in 	= F.relu(torch.cat((pl_1_hidden, m3_hidden), dim = 1))

		# joint layer
		joint_m1_out = self.joint_m1(joint_m1_in)
		joint_m2_out = self.joint_m2(joint_m2_in)
		joint_m3_out = self.joint_m3(joint_m3_in)

		#linear layers on joint out
		fc_m1_out = F.relu(self.fc_m1(joint_m1_out))
		fc_m2_out = F.relu(self.fc_m2(joint_m2_out))
		fc_m3_out = F.relu(self.fc_m3(joint_m3_out))

		# final layers
		out_1 = torch.sigmoid(self.fc_pred_m1(fc_m1_out))
		out_2 = torch.sigmoid(self.fc_pred_m2(fc_m2_out))
		out_3 = torch.sigmoid(self.fc_pred_m3(fc_m3_out))
		
		return  out_1,out_2,out_3


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, vocab_size=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(vocab_size, d_model)
		position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


