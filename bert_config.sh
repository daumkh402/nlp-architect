mkdir ../bert_model
mkdir ../Qbert_model
cp bert_config.json ../bert_model
cp Qbert_config.json ../Qbert_model
cd ../bert_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
mv bert-base-uncased-vocab.txt vocab.txt
mv bert-base-uncased-pytorch_model.bin pytorch_model.bin

cp vocab.txt ../Qbert_model
cp pytorch_model.bin ../Qbert_model




