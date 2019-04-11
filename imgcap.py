import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import os
import json
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from Image_n_cap.model import Complete as Model1
from Image_as_hidden.model import Complete as Model2

IMAGE_WITH_INPUT = 'Image_n_cap/output/Captioner_19.pt'
IWI_I2W = json.load(open('Image_n_cap/id2wordmap.json','r'))
IWI_W2I = json.load(open('Image_n_cap/word2id_map.json','r'))

IMAGE_AS_HIDDEN = 'Image_as_hidden/output/Captioner_20.pt'
IAH_I2W = json.load(open('Image_as_hidden/binderid2wordmap.json','r'))
IAH_W2I = json.load(open('Image_as_hidden/binderword2id_map.json','r'))

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),  # needs to be this size.
    torchvision.transforms.CenterCrop(224),  # needs to be this size.
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),  # the typical.
                                     (0.229, 0.224, 0.225))])


class ImageCaptioningApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.input_img_name = 'No image file selected yet'
        self.model_selected = 'Image with Input'
        self.model1 = Model1(50, len(list(IWI_W2I.keys())), 300, 2)
        self.model1.load_state_dict(torch.load(IMAGE_WITH_INPUT, map_location='cpu'))
        self.model1.eval()
        self.model2 = Model2(50, len(list(IAH_W2I.keys())), 300, 2)
        self.model2.load_state_dict(torch.load(IMAGE_AS_HIDDEN, map_location='cpu'))
        self.model2.eval()
        self.model_in_use = self.model1
        
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.frame1 = tk.Frame(self)
        self.img_name_display = tk.Label(self.frame1)
        self.img_name_display['relief'] = 'sunken'
        self.img_name_display['text'] = self.input_img_name
        self.img_name_display['wraplength'] = 500
        self.img_name_display.pack(side='left')
        self.select_img = tk.Button(self.frame1)
        self.select_img['text'] = 'Choose Image'
        self.select_img['command'] = self.select_img_cmd
        self.select_img.pack(side='right')
        self.frame1.pack(side='top')

        self.frame2 = tk.Frame(self)
        self.var = tk.IntVar()
        self.model_choice1 = tk.Radiobutton(self.frame2)
        self.model_choice1['text'] = 'Image with Input'
        self.model_choice1['command'] = self.select_model
        self.model_choice1['variable'] = self.var
        self.model_choice1['value'] = 1
        self.model_choice1.pack(side='left')
        self.model_choice2 = tk.Radiobutton(self.frame2)
        self.model_choice2['text'] = 'Image as Hidden'
        self.model_choice2['command'] = self.select_model
        self.model_choice2['variable'] = self.var
        self.model_choice2['value'] = 2
        self.model_choice2.pack(side='right')
        self.var.set(1)
        self.frame2.pack(side='top')

        self.img_display = tk.Label(self)
        self.img_display.pack(side='top')

        self.target_display = tk.Label(self)
        self.target_display.pack(side='top')
        self.pred_display = tk.Label(self)
        self.pred_display.pack(side='top')

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")
        

    def select_img_cmd(self):
        self.input_img_name = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(
            ('png files', '*.png'), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.img_name_display['text'] = self.input_img_name
        self.target_display['text'] = ''
        self.pred_display['text'] = ''
        self.img = Image.open(self.input_img_name)
        
        
        self.processing = test_transform(self.img.convert('RGB'))
        self.processing = self.processing.unsqueeze(0)
        selection = self.var.get()
        if selection == 1:
            output, mixedoutput = self.model_in_use.caption(self.processing, 30)
            topcaption = []
            for i in output:
                topcaption.append(IWI_I2W[str(int(i))])
            samplecaption = []
            for i in mixedoutput:
                samplecaption.append(IWI_I2W[str(int(i))])
        else:
            trigger_input = IAH_W2I["<START>"]
            trigger_input = torch.tensor(trigger_input)
            output, mixedoutput = self.model_in_use.caption(trigger_input, self.processing, 30)
            topcaption = []
            for i in output:
                topcaption.append(IAH_I2W[str(int(i))])
            samplecaption = []
            for i in mixedoutput:
                samplecaption.append(IAH_I2W[str(int(i))])
        topstring = ''
        samplestring = ''
        for element in topcaption:
            if element == '<STOP>':
                break
            if element != "<START>":
                topstring += '{} '.format(element)
        for element in samplecaption:
            if element == '<STOP>':
                break
            if element != "<START>":
                samplestring += '{} '.format(element)
        baseheight = 500
        wpercent = (baseheight/float(self.img.size[1]))
        wsize = int((float(self.img.size[0])*float(wpercent)))
        self.img = self.img.resize((wsize,baseheight), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)
        self.img_display['image'] = self.img

        self.target_display['text'] = 'Using Top Word Prediction: {}'.format(topstring)
        self.pred_display['text'] = 'Using Sampling Prediction: {}'.format(samplestring)
        return

    def select_model(self):
        self.img_display['image'] = None
        self.target_display['text'] = ''
        self.pred_display['text'] = ''
        selection = self.var.get()
        if selection == 1:
            self.model_selected = 'Image with Input'
            self.model_in_use = self.model1
        else:
            self.model_selected = 'Image as Hidden'
            self.model_in_use = self.model2
        return


        


root = tk.Tk()
root.title('VOC Demo')
root.geometry('1080x720')
app = ImageCaptioningApplication(master=root)
app.mainloop()