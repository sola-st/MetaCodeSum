import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fnn

import os
import codecs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib

from index import Index
from corpus import Dataset
from model import TRN_Meta,LSTM_Meta
import util as ut


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'--source', help="Path of all training data.", required=True)
parser.add_argument(
	'--destination', help="Path of all destination folder.", required=True)
parser.add_argument(
	'--model_type', help="Select the model. Valid Options: [ lstm, trn] ", required=True)
parser.add_argument(
	'--epoch',  help="num epochs to train default 20 ", required=False,default=5)
parser.add_argument(
	'--head_count',  help="num attention head for transformer model default 5 ", required=False,default=5)
parser.add_argument(
	'--layers',  help="num layers for each transformer default 3 ", required=False,default=3)
parser.add_argument(
	'--ff_count',  help="size of final layer on transformer encoder ", required=False,default=200)
parser.add_argument(
	'--lr',  help="learning rate default .0001 ", required=False,default=.0001)
parser.add_argument(
	'--retrain', help="load existing model and retrain", action='store_true', required=False)
parser.add_argument(
	'--model', help="Path to trained model.", required=False, default=None)
parser.add_argument(
	'--train_filter', help="use subset filtered train file", action='store_true', required=False)
parser.add_argument(
	'--valid_filter', help="use subset filtered valid file", action='store_true', required=False)
parser.add_argument(
	'--test_filter', help="use subset filtered test file", action='store_true', required=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,training_generator,validation_generator,log,loss_type = ""):
	model.to(device)
	model.train()
	
	losses = []
	trn_accs = []
	trn_rocs = []
	trn_hmm_losses = []

	eval_accs = []
	eval_rocs = []
	eval_losses = []
	train_acc = 0
	
	
	regularization = torch.tensor(0.).to(device)
	optimizer = optim.Adam(model.parameters(), lr=ut.lr, betas=(0.9, 0.999),weight_decay=1e-7)
	criterion = nn.BCELoss()
	
	header = "epoch,train_attend,train_ast,train_trn,trn_acc,loss,eval_attend,eval_ast,eval_trn,eval_acc,eval_loss\n"
	
	log.write(header)
	for epoch in range(ut.max_epochs):
		
		target_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
		pred_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
		joint_target = []
		joint_predictions = []
		epoch_loss=0		
		
		for count,data in enumerate(training_generator):
			batch_targets 	= []
			batch_preds 	= []
			body_arr 		= data["body"].to(device)
			code_model_arr 	= data[ut.model_names[0]].to(device)
			seq_model_arr 	= data[ut.model_names[1]].to(device)
			trn_model_arr 	= data[ut.model_names[2]].to(device)

			# input preparation
			input_dict = {"code_tokens": body_arr, ut.model_names[0]:code_model_arr,ut.model_names[1]:seq_model_arr,ut.model_names[2]:trn_model_arr} 

			target1 = data['label1'].to(device)
			target2 = data['label2'].to(device)
			target3 = data['label3'].to(device)
			
			# gold label manipulation 
			add_joint_res(joint_target,[target1,target2,target3],ut.batch_size)

			add_to_arr(target_map,ut.model_names[0],target1)
			add_to_arr(target_map,ut.model_names[1],target2)
			add_to_arr(target_map,ut.model_names[2],target3)
			
			optimizer.zero_grad()

			pl1,pl2,pl3 = model(input_dict)
			
			# predicted label manipulation
			add_joint_res(joint_predictions,[pl1,pl2,pl3],ut.batch_size,True)

			add_to_arr(pred_map,ut.model_names[0],pl1,True)
			add_to_arr(pred_map,ut.model_names[1],pl2,True)
			add_to_arr(pred_map,ut.model_names[2],pl3,True)
			
			batch_targets = torch.stack([target1,target2,target3]).to(device)
			batch_preds = torch.stack([pl1,pl2,pl3]).to(device)
			
			# computing loss on individual level and overall aswell.
			loss1 		= criterion(pl1,target1)
			loss2 		= criterion(pl2,target2)
			loss3 		= criterion(pl3,target3)	
			joint_loss 	= criterion(batch_preds,batch_targets)
			
			cmb_loss = joint_loss + loss1 + loss2 + loss3
			cmb_loss.backward()
			optimizer.step()

			epoch_loss += cmb_loss.item()
			
			
		losses.append(epoch_loss)
		
		# results book keeping
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[0])
		m1_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[1])
		m2_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[2])
		m3_acc=round(accuracy_score(target_arr,pred_arr),2)

		eval_m1_acc,eval_m2_acc,eval_m3_acc,eval_acc,eval_loss = evaluate(model,validation_generator,train_log,criterion,False)

		# accuracy of predicted labels on overall predictions
		trn_acc = accuracy_score(joint_target,joint_predictions)
		# append scores to epoch 
		trn_accs.append(trn_acc)
		eval_accs.append(eval_acc)
		eval_losses.append(eval_loss)
		
		# printing results to screen
		print("{},{},{},{},{},{},{},{},{},{},{}".format(epoch,m1_acc,m2_acc,m3_acc,round(trn_acc,2),round(epoch_loss/(epoch+1),2),eval_m1_acc,eval_m2_acc,eval_m3_acc,eval_acc,round(eval_loss/(epoch+1),2)))
		
		
		log.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,m1_acc,m2_acc,m3_acc,round(trn_acc,2),round(epoch_loss/(epoch+1),2),eval_m1_acc,eval_m2_acc,eval_m3_acc,eval_acc,round(eval_loss/(epoch+1),2)))
		
		# save per epoch models
		torch.save(model.state_dict(), args.destination+"/model_"+str(epoch)+".pt")

	return losses,trn_accs,eval_accs,eval_losses

def evaluate(model, text_generator,log,criterion = None,is_test = True):
	
	target_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
	pred_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
	joint_target = []
	joint_predictions = []
	eval_scr = 0
	eval_loss = 0
	if is_test:
		model.eval()
	with torch.no_grad():
		batch_id =1
		for data in text_generator:
			body_arr = data["body"].to(device)
			code_model_arr = data[ut.model_names[0]].to(device)
			seq_model_arr = data[ut.model_names[1]].to(device)
			trn_model_arr = data[ut.model_names[2]].to(device)

			input_dict = {"code_tokens": body_arr, ut.model_names[0]:code_model_arr,ut.model_names[1]:seq_model_arr,ut.model_names[2]:trn_model_arr} 
			
			target1 = data['label1'].to(device)
			target2 = data['label2'].to(device)
			target3 = data['label3'].to(device)
			add_joint_res(joint_target,[target1,target2,target3],ut.eval_batch_size)
			add_to_arr(target_map,ut.model_names[0],target1)
			add_to_arr(target_map,ut.model_names[1],target2)
			add_to_arr(target_map,ut.model_names[2],target3)

			pl1,pl2,pl3 = model(input_dict)
			if not is_test:
				# compute eval loss
				batch_targets = torch.stack([target1,target2,target3]).to(device)
				batch_preds = torch.stack([pl1,pl2,pl3]).to(device)

				loss1 		= criterion(pl1,target1)
				loss2 		= criterion(pl2,target2)
				loss3 		= criterion(pl3,target3)	
				joint_loss 	= criterion(batch_preds,batch_targets)
								
				cmb_loss = joint_loss + loss1 + loss2 + loss3
				eval_loss += cmb_loss.item()


			add_joint_res(joint_predictions,[pl1,pl2,pl3],ut.eval_batch_size,True)
			add_to_arr(pred_map,ut.model_names[0],pl1,True)
			add_to_arr(pred_map,ut.model_names[1],pl2,True)
			add_to_arr(pred_map,ut.model_names[2],pl3,True)

		# results book keeping
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[0])
		m1_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[1])
		m2_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[2])
		m3_acc=round(accuracy_score(target_arr,pred_arr),2)
		eval_acc = accuracy_score(joint_target,joint_predictions)
		
		print("{},{},{},{}".format(m1_acc,m2_acc,m3_acc,eval_acc))
		log.write("{},{},{},{}".format(m1_acc,m2_acc,m3_acc,eval_acc))
		
	
	return m1_acc,m2_acc,m3_acc,eval_acc,eval_loss

def get_lbl_arr(target_map,pred_map, model_name):
	target_arr = []
	pred_arr = []
	for idx in range(len(target_map[model_name])):
		target_lbl,_ = target_map[model_name][idx]
		target_arr.append(target_lbl)

		pred_lbl,_ = pred_map[model_name][idx]
		pred_arr.append(pred_lbl)
	return target_arr,pred_arr

def plot_loss(destination,train_loss,val_loss,lbl):
	matplotlib.style.use('ggplot')
	plt.figure(figsize=(10, 7))
	plt.plot(train_loss, color='orange', label='train loss')
	plt.plot(val_loss, color='blue', label='train_'+lbl)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(destination)

def plot_scr(destination,trn_scr,val_scr,lbl):
	matplotlib.style.use('ggplot')
	plt.figure(figsize=(10, 7))
	plt.plot(trn_scr, color='blue', label='train_'+lbl)
	plt.plot(val_scr, color='green', label='val_'+lbl)
	plt.xlabel('Epochs')
	
	plt.legend()
	plt.savefig(destination)
	
def __pred_prob_label(elm):
	"""get prediction label from probability """
	if elm >= 0.5:
		return 1.0,elm
	else:
		return 0.0,elm

def add_to_arr(data_map,m_name, to_add,is_pred=False):
	for elms in to_add:
		for elm in elms:
			val = elm.item()
			conf = 0.0
			if is_pred:
				val,conf = __pred_prob_label(elm.item())
			data_map[m_name].append((val,conf))

def add_sep_res(data_map,m_name, to_add,index_loc,is_pred=False):
	for elms in to_add:
		elm = elms[index_loc]
		
		val = elm.item()
		
		if is_pred:
			val,conf = __pred_prob_label(elm.item())
		data_map[m_name].append((val,conf))

def add_joint_res(res_arr,to_add,size,is_pred = False):
	
	
	for indx in range(size):
		conv_arr = []
		pred_lbl = []
		pred_lbl.append(to_add[0][indx])
		pred_lbl.append(to_add[1][indx])
		pred_lbl.append(to_add[2][indx])
		#print(pred_lbl)
		for elm in pred_lbl:
			val = elm.item()
			if is_pred:
				val,_ = __pred_prob_label(elm.item())
			conv_arr.append(val)
		res_arr.append(conv_arr)


if __name__ == '__main__':
	args = parser.parse_args()
	# arg features
	ut.max_epochs = int(args.epoch)
	ut.gamma1 = float(args.gamma1)
	ut.gamma2 = float(args.gamma2)
	ut.lr = float(args.lr)
	ut.head_count =int(args.head_count)
	ut.encoder_layer =int(args.layers)
	ut.ff_count =int(args.ff_count)
	print(args.loss_type)
	print(args.model_type)
	log_path = args.destination
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	model_feat = args.destination.split("/")[-1]
	train_log = codecs.open(log_path+"/train_epoch_log.csv", "a",encoding='utf-8')
	eval_log = codecs.open(log_path+"/eval_log.txt", "a",encoding='utf-8')

	# loading embeddings
	code_embedding,code_ft_model = ut.get_embedding(args.source+"embeddings/cb_code.vec")
	# init index
	code_index = Index()
	code_index.add_word_from_embedding(code_ft_model)

	print("loading nl embedding\n") 
	nl_embedding,nl_ft_model = ut.get_embedding(args.source+"embeddings/cb_nl.vec")
	# init index
	print("init nl index")
	nl_index = Index()
	nl_index.add_word_from_embedding(nl_ft_model)
	
	print("init model\n")
	model = None

	if args.model_type == "trn":
		model = TRN_Meta(code_embedding,nl_embedding, code_index.vocab_size(),nl_index.vocab_size(),len(ut.label_index))
	else:
		model = LSTM_Meta(code_embedding,nl_embedding, code_index.vocab_size(),nl_index.vocab_size(),len(ut.label_index))

	# init corpus
	training_generator = None

	if args.retrain and args.model is not None:
		model.load_state_dict(torch.load(args.model))
	print("data init\n") 
	train_file = "meta_train.json"

	if args.train_filter:
		train_file = "filtered_meta_train.json"

	
	train_dataset = Dataset(args.source+train_file,code_index,nl_index)
	training_generator = torch.utils.data.DataLoader(train_dataset, **ut.params)
	
	
	valid_file = "meta_valid.json"
	if args.valid_filter:
		valid_file = "filtered_meta_valid.json"
	
	validation_set = Dataset(args.source+valid_file,code_index,nl_index)
	validation_generator = torch.utils.data.DataLoader(validation_set, **ut.eval_params)

	test_file = "test.json"
	if args.test_filter:
		test_file = "filtered_test.json"
	test_set = Dataset(args.source+test_file,code_index,nl_index)
	test_generator = torch.utils.data.DataLoader(test_set, **ut.eval_params)

	print("dumping config file")
	config_file = codecs.open(log_path+"/config_file.csv", "w",encoding='utf-8')
	config_file.write("model,{}\n".format(args.model_type))
	config_file.write("train_file,{}\n".format(train_file))
	config_file.write("valid_file,{}\n".format(valid_file))
	config_file.write("epochs,{}\n".format(ut.max_epochs))
	config_file.write("lr,{}\n".format(ut.lr))	
	config_file.write("embedding_dim,{}\n".format(ut.embedding_dim))
	config_file.write("seq_len,{}\n".format(ut.seq_len))
	config_file.write("nl_seq_len,{}\n".format(ut.nl_seq_len))
	config_file.write("batch_size,{}\n".format(ut.batch_size))
	config_file.write("eval_batch_size,{}\n".format(ut.eval_batch_size))
	
	if args.model_type == "trn":
		config_file.write("head,{}\n".format(ut.head_count))
		config_file.write("layer,{}\n".format(ut.encoder_layer))
		config_file.write("feed_dim,{}\n".format(ut.ff_count))
		config_file.write("trn_joint_in_size,{}\n".format(ut.trn_joint_in_size))
		config_file.write("trn_joint_out,{}\n".format(ut.trn_joint_out))
	else:
		config_file.write("hidden_out,{}\n".format(ut.hidden_out))
		config_file.write("code_lstm_out_dim,{}\n".format(ut.code_lstm_out_dim))
		config_file.write("nl_lstm_out_dim,{}\n".format(ut.nl_lstm_out_dim))
		config_file.write("joint_in_size,{}\n".format(ut.joint_in_size))
		config_file.write("joint_out,{}\n".format(ut.joint_out))
	
	config_file.close()

	print("training\n")
	losses,trn_accs,eval_accs,eval_losses = train(model,training_generator,validation_generator,train_log)
	plot_loss(args.destination+"/loss.png",losses,eval_losses,"loss")
	plot_scr(args.destination+"/acc.png",trn_accs,eval_accs,"accuracy")

	print("----------------------------------------")
	print("evaluate \n")
	evaluate(model,test_generator,eval_log)
	
	torch.save(model.state_dict(), args.destination+"/model.pt")
	train_log.close()
	eval_log.close()

	

	
