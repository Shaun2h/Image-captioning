import torch
from PIL import Image
import nltk
nltk.download('punkt')

class data_set(torch.utils.data.Dataset):
    def __init__(self, listy, images_dir, transform, vocab_word, vocab_id):
        self.image_dir = images_dir
        self.transform = transform
        self.data = listy  # our data is a list. how.. quaint.

        self.word2id = vocab_word
        self.id2word = vocab_id  # renamed for better referencing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, caption = self.data[index]
        image = self.transform(
            Image.open(self.image_dir + image_name))  # read the image and transform.
        if image.shape[0] == 1:
            image = image = self.transform(Image.open(self.image_dir + image_name).convert("RGB"))
        caption = nltk.word_tokenize(caption)
        caption_list = []
        caption_list.append(self.word2id["<START>"])  # tape on the start
        for i in caption:
            if i in self.word2id.keys():
                caption_list.append(self.word2id[i])
            else:
                caption_list.append(self.word2id["<UNK>"])  # well. i don't know either.
        caption_list.append(self.word2id["<STOP>"])  # tape on the end
        return image, torch.tensor(caption_list)


def redo_collation(data):
    # Since pytorch's collate function fucks up, I'll reassign it.
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]),
              reverse=True)  # a lamda to sort based off the length of the tuple.
    # Tuple was from image, caption_list of __getitem__
    images, captions = zip(*data)
    # basically now take apart the tuple
    try:
        images = torch.stack(images, dim=0)  # stack across dimension 0
    except RuntimeError:
        for i in images:
            print(i.shape)
    lengths = torch.tensor([len(cap) for cap in captions])  # obtain lengths
    targets = torch.zeros(len(captions), max(lengths))  # create empty array.
    for i in range(len(captions)):
        targets[i, :len(captions[i])] = captions[i]  # attach accordingly. to main array.
        # now you can pad it. after you get the image outputs.
    #     print(lengths.shape)
    #     print(targets.shape)
    #     print(images.shape)
    return images, targets, lengths



# that's for testing. because i forgot about greyscale images..
# for k in range(50):
#     for i in enumerate(val_loader):
#         print("val_pass")
#         print(len(i))
#         break
#     for i in enumerate(train_loader):
#         print(len(i))
#         print("train_pass")
#         break