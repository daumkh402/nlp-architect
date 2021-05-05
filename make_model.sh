


sst2="/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sst-2/lr_2e-5/1/best_dev"
cola="/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/cola/lr_2e-5/1/best_dev"
rte="/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/rte/lr_3e-5/1/best_dev"
stsb="/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sts-b/lr_4e-5/1/best_dev"
mrpc="/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/mrpc/lr_3e-5/1/best_dev" 


#MODEL="mrpc"
#DATA="sst-2"

for MODEL in "mrpc" #"mrpc" "sts-b" "rte" "cola" "sst-2"
do
for DATA in "rte" #"mrpc" "sts-b" "rte" "cola" "sst-2"
do

python make_model.py \
--data  $DATA \
--model $MODEL 


case $MODEL in 
cola) model_dir=$cola;; 	          			
mrpc) model_dir=$mrpc;;       
rte) model_dir=$rte;;        
sts-b) model_dir=$stsb;;             
sst-2) model_dir=$sst2;;                           
esac 

case $DATA in 
cola) data_dir=$cola;; 	          			
mrpc) data_dir=$mrpc;;       
rte) data_dir=$rte;;        
sts-b) data_dir=$stsb;;             
sst-2) data_dir=$sst2;;                           
esac 


label="${data_dir}/labels.txt"
config="${data_dir}/config.json"
vocab="${data_dir}/vocab.txt"

cp  ${label} "/home/imza/ssd/nlp_arch_results/new_models/model_${MODEL}_data_${DATA}"
cp  ${config} "/home/imza/ssd/nlp_arch_results/new_models/model_${MODEL}_data_${DATA}"
cp  ${vocab} "/home/imza/ssd/nlp_arch_results/new_models/model_${MODEL}_data_${DATA}"

done
done
