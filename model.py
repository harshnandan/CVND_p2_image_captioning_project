import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # init super clss
        super(DecoderRNN, self).__init__()
        
        self.embed_dim = embed_size
        self.hidden_dim = hidden_size
        self.vocab_dim = vocab_size
        
        # embedding layer that converts word into vector
        self.word_emedding = nn.Embedding(vocab_size, embed_size)
        
        # define LSTM (input = embedding_dim; output = hidden_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state 
        #self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size):

        # output (num_layers, batch_size, hidden_size)
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        # features and embedding to be stacked together
        features = features.view(len(features), 1, -1)
        
        self.hidden = self.init_hidden(batch_size)
        
        # embedding without the end
        embeddings = self.word_emedding(captions[:, :-1])
        
        # concatenation of feature and embedding
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        
        word_outputs = self.fc(lstm_out)
        
        return word_outputs
        
    def init_hidden(self, batch_size, device):

        return (torch.zeros((1, batch_size, self.hidden_dim), device=device),
                torch.zeros((1, batch_size, self.hidden_dim), device=device))

    def sample(self, inputs, states=None, max_len=20, device=device):

        batch_size = inputs.size(0) 
        self.hidden = self.init_hidden(batch_size, device) 

        predictions = []

        for _ in range(max_len):
            lstm_out, self.hidden = self.lstm(inputs, self.hidden) 
            outputs = self.fc(lstm_out) 
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            predictions.append(max_idx.cpu().numpy()[0].item()) 

            if max_idx == 1:
                break

            inputs = self.word_emedding(max_idx) 
            inputs = inputs.unsqueeze(1) 

        return predictions
    
 