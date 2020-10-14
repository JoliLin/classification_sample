# classification_sample
----
+ python 3.x
+ simple CNN
+ fine-tuning BERT
+ fine-tuning BERT with adapter(Parameter-Efficient Transfer Learning for NLP) 

Reference
----
+ huggingface
+ [Parameter-Effieient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)

Device
----
V100 GPU with 32G RAM * 1

Requirement
----
+ torch
+ numpy 
+ transformers
+ sklearn

Training Detail
----
+ AdamW + warmup + gradient accumulation

Recommended setting and command
----
```
python rt_bert.py --epoch 20 --trainable 3 --lr 1e-4 --batch 16 --accumulative 4
```

Compare
----
|					|exection time		|nums of params(9 layers)	|
|----				|----				|----						|
|Bert(huggingface)	|128s/epoch			|63792386					|	
|Bert with Adapter	|106s/epoch			|806914						|

Notice
----
If run
```
python rt_with_word2vec.py
```
, you have to download the english pretrained model at first.

# Englich pretrained embedding
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

# Chinese pretrained embedding
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
