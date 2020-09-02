# classification_sample
----
+ python 3.x
+ simple CNN
+ fine-tuning BERT

Device
----
V100 GPU with 32G RAM * 1

Requirement
----
+ torch
+ numpy 
+ transformers
+ sklearn

Recommended setting and command
----
```
python rt_bert.py --lr 2e-5 -trainable 9
```

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
