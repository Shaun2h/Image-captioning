import json
import nltk
from json_extract import extract_jsons
nltk.download('punkt')

data_dir = "datasets/"
train_captions = data_dir + "annotations/captions_train2014.json"
val_captions = data_dir + "annotations/captions_val2014.json"

val_data = extract_jsons(open(val_captions))
train_data = extract_jsons(open(train_captions))
vocab_word = {}
vocab_id = {}
idcounter = 0
# # this takes a very very very very long time :c
for item in train_data:
    latest = nltk.word_tokenize(item[1])  # tokenize to obtain special words and characters...
    for i in latest:
        if i not in vocab_word.keys():
            vocab_word[i] = idcounter  # append to the dictionary.
            vocab_id[idcounter] = i
            idcounter += 1  # increment the id counter
for item in val_data:
    latest = nltk.word_tokenize(item[1])  # tokenize to obtain special words and characters...
    for i in latest:
        if i not in vocab_word.keys():
            vocab_word[i] = idcounter  # append to the dictionary.
            vocab_id[idcounter] = i
            idcounter += 1  # increment the id counter
for specials in ("<START>", "<STOP>", "<UNK>"):
    vocab_word[specials] = idcounter
    vocab_id[idcounter] = specials
    idcounter += 1


json.dump(vocab_word, open("word2id_map.json", "w"))  # dump so i never have to wait again
json.dump(vocab_id, open("id2wordmap.json", "w"))  # also pass this to them.

#  you enjoy right.
