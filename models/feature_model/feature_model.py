import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from joblib import dump, load
import json


parser = argparse.ArgumentParser()
parser.add_argument(
	'--source', help="file used for analysis.", required=True)
parser.add_argument(
	'--destination', help="output path for stats.", required=False)
parser.add_argument(
	'--predict', help="load existing model and predict", action='store_true', required=False)
parser.add_argument(
	'--subset_prediction', help="run on subset filtered dataset", action='store_true', required=False)

def prepare_df(source):
	df = None
	df =  pd.read_csv(source)
	
	# extract y columns (depenednt variables)
	y_trn = df["label_trn"]
	y_attend = df["label_attendgru"]
	y_ast = df["label_ast"]
	_ids = df["id"]
	
	# create independent variables
	feature_df = df.copy()
	# drop additional columns from the 
	feature_df= feature_df.drop(["id","bleu_ast","bleu_trn","bleu_attendgru","label_trn","label_attendgru","label_ast",'method_type', 'return_type'],axis=1).copy()
	feature_df= feature_df.replace(np.nan,0)
		
	return {"features": feature_df,"labels": {"trn": y_trn, "attendgru":y_attend, "ast":y_ast},"pred_ids": _ids}


def train(train_set,valid_set,model_name,is_subset_training = False):	
	
	
	if not is_subset_training:
		# due to imbalanced nature of entire dataset we use class weights
		clf = LogisticRegression(random_state=42,solver='saga',max_iter=5000,class_weight={False:1, True:5}) 
	else:
		clf = LogisticRegression(random_state=42,solver='liblinear',max_iter=200) 
	
	# training
	clf.fit(train_set["features"], train_set["labels"][model_name])
	
	# predictions on validation set
	y_pred = clf.predict(valid_set["features"])
	

	print("accuracy on validation set: {}\n".format(accuracy_score(valid_set["labels"][model_name],y_pred)))
	
	return clf

def get_label_confidence(prediction_map,model_name,idx):
	label = prediction_map[model_name]["label"][idx]
	prob_arr = predictions[model_name]["probability"][idx]
	conf_value = prob_arr[0]
	if label:
		conf_value = prob_arr[1]
	return bool(label), round(conf_value,3)


def generate_predictions_file(id_list, predictions):
	prediction_map = {}
	for idx,_id in enumerate(id_list):
		
		labels_map = {}
		confidence_map = {}
		for model_name in ["trn","attendgru","ast"]:
			label, confidence = get_label_confidence(predictions,model_name,idx)
			labels_map[model_name] = label
			confidence_map[model_name] = confidence

		prediction_map[str(_id)] = {"labels":labels_map,"confidence":confidence_map}
			
		
	print(len(prediction_map))	
	return prediction_map

def write_json(data,destination):
	print("writing {}, elements: {}".format(destination,len(data)))
	with open(destination, 'w') as fp:
		json.dump(data, fp)

		
if __name__ == '__main__':
	args = parser.parse_args()
	train_set	= prepare_df(args.source+"ensemble_train/ensemble_train.csv")
	valid_set	= prepare_df(args.source+"ensemble_valid/ensemble_valid.csv")
	test_set	= prepare_df(args.source+"test/test.csv")
	
	# normalization has shown to improve prediction for entire dataset only
	if not args.subset_prediction:
		scaler = StandardScaler()
		train_set["features"] = scaler.fit_transform(train_set["features"])
		valid_set["features"] = scaler.fit_transform(valid_set["features"])
		test_set["features"] = scaler.fit_transform(test_set["features"])
	
	predictions = {"trn":{},"attendgru":{},"ast":{}}
	for model_name in ["trn","attendgru","ast"]:
		if not args.predict:
			model = train(train_set,valid_set,model_name,args.subset_prediction)
		else:
			model = load(args.destination+model_name+".joblib")

		# extract label and confidence score for each prediction
		test_pred = model.predict(test_set["features"])
		test_pred_probs = model.predict_proba(test_set["features"])
		
		predictions[model_name]["label"] = test_pred
		predictions[model_name]["probability"] = test_pred_probs


		print("accuracy on test set: {}\n".format(accuracy_score(test_set["labels"][model_name],test_pred)))
		if not args.predict:
			dump(model, args.destination+model_name+".joblib") 
	
	# extract predictions json for output 
	joint_predictions_map = generate_predictions_file(test_set["pred_ids"],predictions)

	write_json(joint_predictions_map,args.destination+"predictions.json")



	
