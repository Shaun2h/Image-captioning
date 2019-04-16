import matplotlib.pyplot as plt
import json

OUTPUTS_DIRECTORY = 'Image_n_cap/output/'

with open(OUTPUTS_DIRECTORY+'train_loss.json', 'r') as f:
    train_loss = json.load(f)
train_loss.reverse()
with open(OUTPUTS_DIRECTORY+'val_loss.json', 'r') as f:
    val_loss = json.load(f)
val_loss.reverse()
with open(OUTPUTS_DIRECTORY+'val_accuracies.json', 'r') as f:
    val_acc = json.load(f)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss[:-1])

plt.subplot(1,3,2)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(val_loss)

plt.subplot(1,3,3)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(val_acc)

plt.show()