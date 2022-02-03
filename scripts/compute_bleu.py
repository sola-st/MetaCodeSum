# 
# compute blue score for dataset
#
import argparse
import util



parser = argparse.ArgumentParser()
parser.add_argument(
	'--source', help="Path to funcom dataset source files.", required=True)
parser.add_argument(
	'--model', help="Path to model output.", required=True)
parser.add_argument(
	'--subset', help="use subset filtered test file", action='store_true', required=False)

def compute_bleu(refs, preds):
	Ba = corpus_bleu(refs, preds)
	Ba = round(Ba * 100, 2)
	
	B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
	B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
	B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
	B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))
	
	
	B1 = round(B1 * 100, 2)
	B2 = round(B2 * 100, 2)
	B3 = round(B3 * 100, 2)
	B4 = round(B4 * 100, 2)
	
	ret = ''
	ret += ('for %s functions\n' % (len(preds)))
	ret += ('Ba %s\n' % (Ba))	
	ret += ('B1 %s\n' % (B1))
	ret += ('B2 %s\n' % (B2))
	ret += ('B3 %s\n' % (B3))
	ret += ('B4 %s\n' % (B4))
	
	return ret,Ba

def get_results(source_map,pred_map):
	refs = []
	preds = []
	
	for _id, source_commment in  source_map.items():
		if _id in pred_map:
			pred_summ =  pred_map[_id]
			
			preds.append(pred_summ.strip().split())
			refs.append([source_commment.strip().split()])
			
	print("{},{}".format(len(refs),len(preds)))
	ret, Ba = compute_bleu(refs, preds)

	print(ret)

	return Ba


def get_model_results(source_map,pred_map):
	refs = []
	preds = []
	
	for _id, source_commment in  source_map.items():
		if _id in pred_map:
			pred_summ =  pred_map[_id]["comment"]
			
			preds.append(pred_summ.strip().split())
			refs.append([source_commment.strip().split()])
			
	print("{},{}".format(len(refs),len(preds)))
	
	res,ba = compute_bleu(refs, preds)
	print(res)
	return ba



def load_filter(filter_fname):
	"""
	Ids for filter subset
	"""
	id_list = set()
	with open(filter_fname, "r") as f:
		for line in f:
			
			id_list.add(str(line.strip()))
	print("filtered set: {}".format(len(id_list)))
	return id_list


def filter_data(data_map, filtered_keys):	
	filtered_map = {}
	for _id, commment in  data_map.items():
		if _id in filtered_keys:
			filtered_map[_id] = commment
	print(len(filetered_map))
	return filtered_map

if __name__ == '__main__':
	args = parser.parse_args()
	# read 
	source_map = util.read_fncm_files(args.source+"nl.csv")
	prediction_map = util.read_fncm_files(args.model+"meta_summ.csv")



	source_map = filter_source(source_map,args.source+"subset_ids.txt")
	
	if args.subset:
		filtered_keys = load_filter(args.source+"subset_ids.txt")
		source_map = filter_data(source_map,filtered_keys)
		prediction_map = filter_data(prediction_map,filtered_keys)

	
	get_model_results(source_map,prediction_map)
	