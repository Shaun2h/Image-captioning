# Version with gpu.
# frozen VGG

import torch
import torchvision
import nltk
import math
import json
import os
from json_extract import extract_jsons
from torch.nn.utils.rnn import pack_padded_sequence
from dataset_class import data_set
from dataset_class import redo_collation

from model import Complete
from PIL import Image

nltk.download('punkt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "datasets/"
train_captions = data_dir + "annotations/captions_train2014.json"
val_captions = data_dir + "annotations/captions_val2014.json"
train_images_dir = data_dir + "train2014/"
val_images_dir = data_dir + "val2014/"
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # thanks jingyu for dis SIK TIP

#############################################


current_epoch = 0  # if your model training gets interrupted...
total_epochs = 30
embedding_size = 50  # hyper parameters
lr = 0.001  # uh something small.
momentum = 0.9  # the usual
num_layers = 2
num_hidden = 300
batch_size = 8  # from 32 before epoch 8 #from 16 before epoch 15 (cap14\) to 8
loss_fn = torch.nn.CrossEntropyLoss()  # didn't put a softmax layer..
num_workers = 0
decay_rate = 0.9
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(300),  # resized to 300.
    torchvision.transforms.RandomCrop(224),  # needs to be this size.
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     # the typical. no 5 crop because we don't want to wait..
                                     (0.229, 0.224, 0.225))])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),  # needs to be this size.
    torchvision.transforms.CenterCrop(224),  # needs to be this size.
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),  # the typical.
                                     (0.229, 0.224, 0.225))])

#############################################
val_data = extract_jsons(open(val_captions))
train_data = extract_jsons(open(train_captions))
print(val_data[0])  # tuple. ( image file name, caption)

vocab_word = json.load(open("word2id_map.json", "r"))
vocab_id = json.load(open("id2wordmap.json", "r"))


holdy = data_set(val_data, val_images_dir, test_transform, vocab_word, vocab_id)
val_loader = torch.utils.data.DataLoader(holdy, batch_size=batch_size, shuffle=True, num_workers=0,
                                         collate_fn=redo_collation)
holdy = data_set(train_data, train_images_dir, train_transform, vocab_word, vocab_id)
train_loader = torch.utils.data.DataLoader(holdy, batch_size=batch_size, shuffle=True,
                                           num_workers=0, collate_fn=redo_collation)

model = Complete(embedding_size, len(list(vocab_word.keys())), num_hidden, num_layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_rate, last_epoch=-1)
#  torch.save(model.state_dict(), PATH)
#  model = TheModelClass(*args, **kwargs)
model = model.to(device)
#  model.load_state_dict(torch.load(PATH))
Train_loss = []
Val_loss = []
Accuracies = []
train_length = math.ceil(len(train_loader)/batch_size)
val_length = math.ceil(len(val_loader)/batch_size)
print("train length: " + str(train_length))
print("val length: " + str(val_length))


if current_epoch != 0:
    pass
    # Val_loss = json.load(open(output_dir + "val_loss.json"))

    # Accuracies = json.load(open(output_dir + "val_accuracies.json"))
    # Train_loss = json.load(open(output_dir + f"train_loss.json"))
    # model.load_state_dict(torch.load(output_dir + f"Captioner_{current_epoch-1}.pt"))


for epoch_num in range(current_epoch, total_epochs):
    model.train()
    total_loss = 0
    print("Beginning Training")

    for idx, (image, caption, seqlen) in enumerate(train_loader):
        targets = torch.nn.utils.rnn.pack_padded_sequence(caption, seqlen, batch_first=True)[0]
        #         print(image.shape)  # torch.Size([32, 3, 224, 224])
        #         print(caption.shape)  # torch.Size([32, ???])
        #         print(seqlen.shape)  # torch.Size([32])
        # 32 is technically batchsize
        # ??? is the longest tensor's length for captions.
        # feed in caption here.
        if image.shape[0] == 1:
            continue
        optimizer.zero_grad()
        output = model(image.float().to(device), caption.long().to(device),
                       seqlen.float().to(device))
        #         print(output.shape)
        #         print(targets.shape)
        targets = targets.long().to(device)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx % 50 == 0 and idx != 0:
            print("50 Batches done")
    #         break  #uncomment to TEST

    Train_loss.append(total_loss)  # save
    torch.save(model.state_dict(), output_dir + f"Captioner_{epoch_num}.pt")
    json.dump(Train_loss, open(output_dir + f"train_loss.json", "w"))
    print("Training Completed. Train loss, model parameters, dumped.")
    model.eval()
    correct = 0
    total_encountered = 0
    total_loss = 0
    print("Beginning Validation")
    with torch.no_grad():
        for idx, (image, caption, seqlen) in enumerate(val_loader):
            #         print(image.shape)  # torch.Size([32, 3, 224, 224])
            #         print(caption.shape)  # torch.Size([32, ???])
            #         print(seqlen.shape)  # torch.Size([32])
            # return_seq = model.caption(image.float(),seqlen[0])
            # if you're doing sampling.. but we aren't. ignore this.
            targets = torch.nn.utils.rnn.pack_padded_sequence(caption, seqlen, batch_first=True)[0]
            output = model(image.float().to(device), caption.long().to(device),
                           seqlen.float().to(device))
            #         print(output.shape)
            targets = targets.long().to(device)
            loss = loss_fn(output, targets.long())
            total_loss += loss.item()
            _, output = torch.max(output, dim=1)
            #         print("after")
            #         print(output.shape)
            #         print(targets.shape)
            #         print(output)
            #         print(caption)
            for singular in range(len(targets)):
                total_encountered += 1
                if int(targets[singular]) == int(output[singular]):
                    correct += 1
                    # individual comparison of whether it was CORRECT.
            if idx % 50 == 0 and idx != 0:
                print("50 Batches done")
    #         break #uncomment to TEST
    #     break #uncomment to TEST

    Val_loss.append(total_loss)  # save
    Accuracies.append(correct / total_encountered)
    json.dump(Val_loss, open(output_dir + "val_loss.json", "w"))
    json.dump(Accuracies, open(output_dir + "val_accuracies.json", "w"))
    print("Validation Completed. Validation losses and accuracies dumped.")
    scheduler.step()
print("Done")


