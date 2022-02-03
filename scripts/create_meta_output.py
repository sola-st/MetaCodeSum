"""creates a final output file from the label predictions of meta model"""
import json
import argparse
import random
import util

parser = argparse.ArgumentParser()
parser.add_argument(
	'--source', help="Path of all original data.", required=True)
parser.add_argument(
	'--predicted', help="Path meta model predictions file", required=True)
parser.add_argument(
	'--destination', help="Path of all output file.", required=True)

def load_predictions(source):
	result_map = {}
	with open(source, "r") as f:
		data_map = json.load(f)
		print(len(data_map))
		for key,vals in data_map.items():
			_id = str(key)
			if _id not in result_map:
				result_map[_id] = vals
			
	print("read {} items in predictions_map".format(len(result_map)))
	
	return result_map

def load_source(source):
	og_model_map = {}
	with open(source, "r") as f:
		data_list = json.load(f)
		print(len(data_list))
		for data in data_list:
			_id =str(data["_id"])
			if _id not in og_model_map:
				og_model_map[_id] = data				
	#print(og_model_map)		
	print("read {} items in source_map".format(len(og_model_map)))
	return og_model_map

def get_prediction_summaries(source_map,predicted_map):
	pred_summ_map = {}
	for _id, predictions in predicted_map.items():
		pred_summary = "UNK"
		if _id in source_map:
			og_summs 	= source_map[_id]["summary"]
			
			pred_labels = predictions["labels"]
			pred_confs 	= predictions["confidence"]
			candidates= {}
			# if all predictions are false then consider the one with the highest score
			if is_all_false(pred_labels):
				candidates = pred_confs
			else:	
				if pred_labels["attendgru"]:
					candidates["attendgru"] = pred_confs["attendgru"]
				
				if pred_labels["ast"]:
					candidates["ast"] = pred_confs["ast"]

				if pred_labels["trn"]:
					candidates["trn"] = pred_confs["trn"]

			# select the model with the highest confidence
			predicted_key = max(candidates, key=candidates.get)
			pred_summary = og_summs[predicted_key]

		if _id not in pred_summ_map:
			pred_summ_map[_id] = pred_summary
	print("total predicted summaries {}".format(len(pred_summ_map)))
	return pred_summ_map

def is_all_false(lbl_map):
	if not lbl_map["attendgru"]:
		if not lbl_map["ast"]:
			if not lbl_map["trn"]:
				return True
	return False



	

def convert_to_csv(data_map):
	data_arr = []
	
	for key, val in data_map.items():
		row_elm = []
		row_elm.append(key)
		row_elm.append(val)		
		data_arr.append(row_elm)

	return data_arr


if __name__ == '__main__':
	args 		  = parser.parse_args()
	source_map 	  = load_source(args.source)
	predicted_map = load_predictions(args.predicted)
	pred_summ_map = get_prediction_summaries(source_map,predicted_map)

	summ_arr 	  = convert_to_csv(pred_summ_map)
	util.write_csv(summ_arr,[],args.destination+"meta_summ.csv","\t",False)
	
