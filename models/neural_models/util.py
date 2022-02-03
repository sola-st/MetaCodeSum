from gensim.models import KeyedVectors
import csv
import torch

max_epochs = 5
embedding_dim =100

seq_len = 50
nl_seq_len = 20

hidden_out = 50

code_trn_out_dim = 50
code_lstm_out_dim = 50
nl_lstm_out_dim = 10
nl_trn_out_dim = 20

joint_in_size = (code_lstm_out_dim*2)+(nl_lstm_out_dim*2)
trn_joint_in_size = code_lstm_out_dim+nl_trn_out_dim
trn_joint_out = 50
joint_out = 25

batch_size = 25
eval_batch_size = 100
predict_batch_size = 1
gamma1 = 0.3
gamma2 = 0.1
l2 = 1e-6
lr = 1e-4
ff_count = 200
encoder_layer = 3
head_count = 5
params = {'batch_size': batch_size,'shuffle': True,'num_workers': 0,'drop_last': True}
eval_params = {'batch_size': eval_batch_size,'shuffle': False,'num_workers': 0,'drop_last': True}
prediction_params = {'batch_size': predict_batch_size,'shuffle': False,'num_workers': 0,'drop_last': True}

label_index = {"attendgru":0,"ast":1,"trn":2}
index_label = {0:"attendgru",1:"ast",2:"trn"}
target_names = ['attendgru', 'ast', 'trn']
model_names = ['attendgru', 'ast', 'trn']


def get_embedding(path):
	ft_emb = KeyedVectors.load_word2vec_format(path,encoding = 'utf8')
	pad_tensor = torch.zeros(1,embedding_dim)
	oov_tensor = torch.zeros(1,embedding_dim)
	w = torch.cat((pad_tensor,oov_tensor))
	weights = torch.cat((w,torch.FloatTensor(ft_emb.vectors)))
	return weights,ft_emb

def write_index(destination,vocab_map):
	print("writing file {0}".format(destination))
	with open(destination, 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in vocab_map.items():
			writer.writerow([key, value])
	
def __load_index(source,is_i2w = False):
	print("reading file {0}".format(source))
	val_map = {}
	with open(source) as csv_file:
		reader = csv.reader(csv_file)
		val_map = dict(reader)

	cast_map = {}
	if is_i2w:
		for key, val in val_map.items():
			cast_map[int(key)] = val
	else:
		for key, val in val_map.items(): 
			cast_map[int(val)] = key
	return cast_map