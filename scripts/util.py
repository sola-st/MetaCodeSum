import csv
import json

def write_csv(data,header,destination,_delimeter=",", write_header = True):
	print("writing {}, elements: {}".format(destination,len(data)))
	with open(destination,'w') as out:
		csv_out=csv.writer(out,delimiter=_delimeter)
		if write_header and len(header)>0:
			csv_out.writerow(header)
		for row in data:
			csv_out.writerow(row)

def read_fncm_files(source):
	"""
	Reads data from fncm split files are returns a dictionary from it
	"""	
	data_map = {}
	
	# read data and add the tokens to the vocab
	with open(source, "r") as f:
		for line in f:
			_id,_txt = line.split("\t")
			if _id not in data_map:
				data_map[_id] = _txt
	print("{}".format(len(data_map)))
	return data_map


def write_json(data,destination):
	print("writing {}, elements: {}".format(destination,len(data)))
	with open(destination, 'w') as fp:
		json.dump(data, fp)