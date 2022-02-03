"""Generate predictions for Neural Meta learning models"""
import torch
import argparse
import json
import os
import codecs
from sklearn.metrics import accuracy_score


from model import TRN_Meta,LSTM_Meta
from corpus import Dataset
from index import Index
from main import add_joint_res
from main import add_sep_res
from main import __pred_prob_label,add_to_arr,get_lbl_arr
import util as ut

parser = argparse.ArgumentParser()
parser.add_argument(
	'--source', help="Path to source files.", required=True)
parser.add_argument(
	'--model', help="Path to trained model.", required=True)
parser.add_argument(
	'--model_type', help="Select the model. Valid Options: [ lstm, trn] ", required=True)
parser.add_argument(
	'--destination', help="Path to folder where the filtered files will be exported.", required=True)
parser.add_argument(
	'--head_count',  help="num attention head default 5 ", required=False,default=5)
parser.add_argument(
	'--layers',  help="num layers for each transformer default 3 ", required=False,default=3)
parser.add_argument(
	'--ff_count',  help="size of final layer on transformer encoder ", required=False,default=200)
parser.add_argument(
	'--test_filter', help="use subset filtered test file", action='store_true', required=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, text_generator,log):
	target_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
	pred_map = {ut.model_names[0]:[],ut.model_names[1]:[],ut.model_names[2]:[]}
	prediction_map = {}
	joint_target = []
	joint_predictions = []
	eval_scr = 0
	
	model.to(device)
	model.eval()
	with torch.no_grad():
		batch_id =1
		for count,data in enumerate(text_generator):
			try:
				
				body_arr = data["body"].to(device)
				code_model_arr = data[ut.model_names[0]].to(device)
				seq_model_arr = data[ut.model_names[1]].to(device)
				trn_model_arr = data[ut.model_names[2]].to(device)

				input_dict = {"code_tokens": body_arr, ut.model_names[0]:code_model_arr,ut.model_names[1]:seq_model_arr,ut.model_names[2]:trn_model_arr} 
				
				target1 = data['label1'].to(device)
				target2 = data['label2'].to(device)
				target3 = data['label3'].to(device)
				add_joint_res(joint_target,[target1,target2,target3],ut.predict_batch_size)
				add_to_arr(target_map,ut.model_names[0],target1)
				add_to_arr(target_map,ut.model_names[1],target2)
				add_to_arr(target_map,ut.model_names[2],target3)

				pl1,pl2,pl3 = model(input_dict)
				
				_id = data["_id"][0]
				
				add_predictions(prediction_map,_id,[pl1,pl2,pl3])

				add_joint_res(joint_predictions,[pl1,pl2,pl3],ut.predict_batch_size,True)
				add_to_arr(pred_map,ut.model_names[0],pl1,True)
				add_to_arr(pred_map,ut.model_names[1],pl2,True)
				add_to_arr(pred_map,ut.model_names[2],pl3,True)
			except Exception as e:
				print(e)
			
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[0])
		m1_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[1])
		m2_acc=round(accuracy_score(target_arr,pred_arr),2)
		
		target_arr,pred_arr = get_lbl_arr(target_map,pred_map,ut.model_names[2])
		m3_acc=round(accuracy_score(target_arr,pred_arr),2)

		eval_acc = round(accuracy_score(joint_target,joint_predictions),2)
		
		
		print("{},{},{},{}".format(m1_acc,m2_acc,m3_acc,eval_acc))
		log.write("m1_Acc,m2_acc,m3_acc,eval_acc\n")
		log.write("{},{},{},{}\n".format(m1_acc,m2_acc,m3_acc,eval_acc))
				
		
		return prediction_map


def write_json(data,destination):
	print("writing {}, elements: {}".format(destination,len(data)))
	with open(destination, 'w') as fp:
		json.dump(data, fp)

def add_predictions(prediction_map,_id,predicted_labels):
	if _id not in prediction_map:
		prediction_map[_id] = {}
		prediction_map[_id]["labels"] = {ut.model_names[0]: False,ut.model_names[1]: False,ut.model_names[2]: False}
		prediction_map[_id]["confidence"] = {ut.model_names[0]: 0.0,ut.model_names[1]: 0.0,ut.model_names[2]: 0.0}
	lbl,conf= __pred_prob_label(predicted_labels[0][0].item())
	if(lbl == 1.0):
		prediction_map[_id]["labels"][ut.model_names[0]] = True
	prediction_map[_id]["confidence"][ut.model_names[0]] = conf
	
	lbl,conf= __pred_prob_label(predicted_labels[1][0].item())
	if(lbl == 1.0):
		prediction_map[_id]["labels"][ut.model_names[1]] = True
	prediction_map[_id]["confidence"][ut.model_names[1]] = conf

	lbl,conf= __pred_prob_label(predicted_labels[2][0].item())
	if(lbl == 1.0):
		prediction_map[_id]["labels"][ut.model_names[2]] = True
	prediction_map[_id]["confidence"][ut.model_names[2]] = conf

if __name__ == '__main__':
	args = parser.parse_args()
	log_path = args.destination
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	ut.head_count = int(args.head_count)
	ut.encoder_layer =int(args.layers)
	ut.ff_count =int(args.ff_count)
	pred_log = codecs.open(log_path+"/pred.txt", "w",encoding='utf-8')

	# loading embeddings
	print("loading embedding\n") 
	
	print("loading codecs embedding\n") 
	code_embedding,code_ft_model = ut.get_embedding(args.source+"embeddings/cb_code.vec")
	# init index
	print("init index")
	code_index = Index()
	code_index.add_word_from_embedding(code_ft_model)

	print("loading nl embedding\n") 
	nl_embedding,nl_ft_model = ut.get_embedding(args.source+"embeddings/cb_nl.vec")
	# init index
	print("init nl index")
	nl_index = Index()
	nl_index.add_word_from_embedding(nl_ft_model)

	#load model
	print("init model\n")
	if args.model_type == "lstm":
		model = LSTM_Meta(code_embedding,nl_embedding, code_index.vocab_size(),nl_index.vocab_size(),len(ut.label_index))
	
	else:	
		model = TRN_Meta(code_embedding,nl_embedding, code_index.vocab_size(),nl_index.vocab_size(),len(ut.label_index))

	print("model init")
	model.load_state_dict(torch.load(args.model))
	print("model loaded")
	test_file = "test.json"
	if args.test_filter:
		test_file = "filtered_test.json"

	test_set = Dataset(args.source+test_file,code_index,nl_index)
	test_generator = torch.utils.data.DataLoader(test_set, **ut.prediction_params)
	print("data loading")

	prediction_map = predict(model,test_generator,pred_log)
	
	fname = "/predictions" 
	write_json(prediction_map,args.destination+fname+".json")

