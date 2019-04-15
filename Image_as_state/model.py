import torch
import torchvision


class Complete(torch.nn.Module):
    def __init__(self, embed_size, total_words, num_hidden, num_layers):
        super(Complete, self).__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = torchvision.models.vgg19_bn(pretrained=True)
        self.resnet.classifier[6] = torch.nn.Linear(self.resnet.classifier[6].in_features,
                                                    num_hidden)  # change to different thing
        for param in self.resnet.features.parameters():
            param.requires_grad = False
        for param in self.resnet.classifier[6].parameters():
            param.requires_grad = True
        # self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features,embed_size)
        # output to same dim as embeddings.
        self.batch_norm = torch.nn.BatchNorm1d(
            num_hidden)  # for smoothness. Also 1d because its 3dim i fucked this up :C
        self.embed = torch.nn.Embedding(total_words,
                                        embed_size)  # embedding, takes in the number of words...
        # https://www.quora.com/How-do-I-determine-the-number-of-dimensions-for-word-embedding
        # essentially hyper parameter
        self.num_layers = num_layers
        self.hidden = num_hidden

        self.lstm = torch.nn.LSTM(embed_size, num_hidden, num_layers,
                                  batch_first=True)  # ensure batch,seq,word
        self.last = torch.nn.Linear(num_hidden,
                                    total_words)
        # need to output same amount of words as vocab size.
        # return here.

    def forward(self, image, caption, length):
        out1 = self.resnet(image)  # resnet the image, getting your outputs.. linear was .. edited
        # maybe i should only toggle the linear layer? leave resnet itself untouched...?
        #         print(caption.shape)
        out2 = self.batch_norm(
            out1)  # batchnorm result torch.Size([32, 50]) # this is the result of your resnet
        #         print("original caption shape" + str(caption.shape))
        caption = self.embed(
            caption)
        # embed the linear output. torch.Size([32, 19, 50]) # these are your captions..
        # you should concat along dimension1. but you need to unsqueeze batchnorm to 32,1,50
        #         out2 = out2.unsqueeze(1)
        #         out3 = torch.cat((out2,caption),dim = 1)
        # you concatenate along the dim 1 because dim 0 is batches.
        hiddens = self.initHidden(image.shape[0])
        #         out2 = out2.unsqueeze(0)
        holding_list = []
        for i in range(self.num_layers):
            holding_list.append(out2)
        hiddens[0] = torch.stack(holding_list, dim=0)  # stack to make images fed into network
        out3 = caption
        #         print("Hidden:")
        #         print(hiddens[0].shape)
        #         print("cellstate:")
        #         print(hiddens[1].shape)
        #         print("out3.shape:")
        #         print(out3.shape)
        # batch,seq, whatever # out3 shape = batch,longestseq, embedding size
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(out3, length, batch_first=True)
        # essentially...
        # pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths)
        #         try:
        out4, cell_state_nonsense = self.lstm(lstm_input, hiddens)
        #         except RuntimeError as e:
        #             print("out2.shape : "+ str(out2.shape))
        #             print("caption.shape after embed: "+ str(caption.shape))
        #             print("length.shape : "+ str(length.shape))
        #             print("out3.shape : "+ str(out3.shape))
        #             print(str(e))
        #             print(length)
        #             print(caption)
        #             output, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #                                                           lstm_input, batch_first=True)

        # so here's the funny thing. you apparently put both image
        # and caption in as inputs for training after
        # concatenation.
        # i don't get the math
        # out4 = batch,seq, hidden_dims except you can't see dis shape since it is packed.
        #         output, _ = torch.nn.utils.rnn.pad_packed_sequence(out4, batch_first=True)
        out5 = self.last(out4[0])
        return out5

    # https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
    def caption(self, trigger, image,
                max_length):  # ensure image is indeed of a single batch only. i.e. [1, 3,255]!
        # fuckit you get it
        words = []
        mixedwords = []
        out1 = self.resnet(image)  # process all images
        out2 = self.batch_norm(out1)
        # input to lstm needs to be [1,1,50]
        trigger = self.embed(trigger).unsqueeze(0).unsqueeze(0)
        #         print(trigger.shape)
        hiddens = self.initHidden(1)
        #         out2 = out2.unsqueeze(0)
        holding_list = []
        for i in range(self.num_layers):
            holding_list.append(out2)
        hiddens[0] = torch.stack(holding_list, dim=0)  # stack to make  images fed into network
        for i in range(
                max_length):  # maxlength is a hyper parameter here, since you don't know when it's optimal to cut...
            # better extra then nothing to look at at all though...
            #             print(out2.shape)
            #             print(hiddens[0].shape)
            #             print(hiddens[1].shape)
            out3, hiddens = self.lstm(trigger,
                                      hiddens)  # amazing how you actually don't put in anything. i don't get math
            out4 = self.last(out3)
            out4 = out4.squeeze(0)
            #             print(out4.shape) [1, vocabsize]
            #             if greed:
            _, sampled = torch.max(out4,
                                   dim=1)  # look at dimension 1, since now it's just 1, vocab
            #             print(sampled.shape)
            #             sampled = int(torch.distributions.categorical.Categorical(torch.exp(out4.squeeze(0))).sample())
            trigger = self.embed(sampled.unsqueeze(0).long())
            #             print(trigger.shape)
            # greedy method. try cat?
            # value,indices
            #             else:
            sampledz = int(
                torch.distributions.categorical.Categorical(torch.exp(out4.squeeze(0))).sample())
            # catty
            words.append(sampled)
            mixedwords.append(sampledz)
        #             out2 = self.embed(sampled).squeeze(1) # and then you put this in as the next input. yes. the word.
        # i don't get it either.

        return words, mixedwords  # batch_size, max_seq_length here

    def initHidden(self, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return ([torch.zeros(self.num_layers, batch_size, self.hidden).to(device),
                 torch.zeros(self.num_layers, batch_size, self.hidden).to(device)])
