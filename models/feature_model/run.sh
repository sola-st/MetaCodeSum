
DESTINATION_PATH=../../results
DESTINATION_DIR=${DESTINATION_PATH}/"$(date +%d-%m-%Y-%H-%M-%S-%N)"

echo $DESTINATION_DIR
if [ ! -d $DESTINATION_DIR ]; then
  mkdir -p $DESTINATION_DIR;
fi

MODEL_DESTINATION=${DESTINATION_DIR}/model.pt
DATA_SOURCE=../../data/meta_learning/

META_LBL_PREDICTION=${DESTINATION_DIR}/predictions.json
RES_DESTINATION=${DESTINATION_DIR}/res/
if [ ! -d $RES_DESTINATION ]; then
  mkdir -p $RES_DESTINATION;
fi

echo "running training"
python3 feature_model.py --source $DATA_SOURCE --destination $DESTINATION_DIR
echo "done training"
echo "------------------------------------------------"


echo "selecting  candidate summaries from meta predictions"
python3 ../../scripts/create_meta_output.py --source ${DATA_SOURCE}/filtered_test.json --predicted $META_LBL_PREDICTION --destination $RES_DESTINATION
echo "------------------------------------------------"

echo "compute corpus bleu"
python3 ../../scripts/compute_bleu.py --source ${DATA_SOURCE}/source/ --meta $RES_DESTINATION --subset> ${RES_DESTINATION}/subset_corpus_bleu.txt

echo "done"




