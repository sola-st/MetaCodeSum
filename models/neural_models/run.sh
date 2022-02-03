# script to run neural meta-learning end to end
DESTINATION_PATH=../../results
DESTINATION_DIR=${DESTINATION_PATH}/"$(date +%d-%m-%Y-%H-%M-%S-%N)"

echo $DESTINATION_DIR
if [ ! -d $DESTINATION_DIR ]; then
  mkdir -p $DESTINATION_DIR;
fi

MODEL_DESTINATION=${DESTINATION_DIR}/model.pt
DATA_SOURCE=../../data/

META_LBL_PREDICTION=${DESTINATION_DIR}/predictions.json
RES_DESTINATION=${DESTINATION_DIR}/res/
if [ ! -d $RES_DESTINATION ]; then
  mkdir -p $RES_DESTINATION;
fi

#-------------------------- NOTE: ------------------------------
# There are multiple options to train and test the neural models. Details of which can be found in main.py examples to which are as follow

# 1. To train different models specify model type: valid options [lstm, trn]
# 2. To train and test models on filtered subsets specify [--train_filter,--valid_filter,--test_filter] flags
#--------------------------------------------------------

echo "running training"

python3 main.py --source ${DATA_SOURCE}/meta_learning/neural_network/ --destination $DESTINATION_DIR  --epoch 1 --model_type lstm  --lr 1e-4 --train_filter --valid_filter --test_filter

echo "done training"
echo "------------------------------------------------"

echo "generating predictions labels for trained model"
python3 get_meta_predictions.py --source ${DATA_SOURCE}/meta_learning/neural_network/ --destination $DESTINATION_DIR  --model $MODEL_DESTINATION --model_type lstm --test_filter

echo "done with prediction labels"

echo "------------------------------------------------"

echo "selecting  candidate summaries from meta predictions"
python3 ../../scripts/create_meta_output.py --source ${DATA_SOURCE}/meta_learning/neural_network/filtered_test.json --predicted $META_LBL_PREDICTION --destination $RES_DESTINATION
echo "------------------------------------------------"

echo "compute corpus bleu"
python3 ../../scripts/compute_bleu.py --source ${DATA_SOURCE}/source/ --meta $RES_DESTINATION --subset > ${RES_DESTINATION}/subset_corpus_bleu.txt

echo "done"





