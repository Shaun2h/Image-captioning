import torch
import json
import torchvision
from PIL import Image
from model import Complete
import sys
#############################################
if len(sys.argv) != 3:
    print("python sample.py <model> <image>")
    quit()
image_arg = sys.argv[2]
model_target = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 50  # hyper parameters
num_layers = 2
num_hidden = 300

vocab_word = json.load(open("binderword2id_map.json", "r"))
vocab_id = json.load(open("binderid2wordmap.json", "r"))

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),  # needs to be this size.
    torchvision.transforms.CenterCrop(224),  # needs to be this size.
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),  # the typical.
                                     (0.229, 0.224, 0.225))])

#############################################
model = Complete(embedding_size, len(list(vocab_word.keys())), num_hidden, num_layers)

model = model.to(device)
model.load_state_dict(torch.load(model_target, map_location=device))
model.eval()

image = test_transform(Image.open(image_arg))
image = image.unsqueeze(0).to(device)
trigger_input = vocab_word["<START>"]
trigger_input = torch.tensor(trigger_input).to(device)
output, mixedoutput = model.caption(trigger_input, image, 30)
caption = []
print("Top values")
for i in output:
    caption.append(vocab_id[str(int(i))])
print(caption)
caption = []
print("Categorical sampled")
for i in mixedoutput:
    caption.append(vocab_id[str(int(i))])
print(caption)
