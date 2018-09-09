import torch
import torch.nn as nn
from torch.autograd import Variable

from neural_srl.util.utils import *

class DecoderLinear(nn.Module):

    def __init__(self, input_dimension, tag_to_ix, input_dropout_p=0):
        
        super(DecoderLinear, self).__init__()
        
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.dropout = nn.Dropout(input_dropout_p)
        self.hidden2tag = nn.Linear(input_dimension, self.tagset_size)
        self.ignore = -1
        self.loss_function = nn.CrossEntropyLoss(ignore_index= self.ignore, size_average=False)
    
    def decode(self, input_var, wordslen, usecuda=True, score_only= False):
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        
        score = nn.functional.softmax(features, dim = 2)
        if score_only:
            return score.data.cpu().numpy()
        
        tag_seq_t = torch.max(score, dim = 2)[1].data.cpu().numpy().tolist()
        tag_seq = [ts[:wordslen[i]] for i,ts in enumerate(tag_seq_t)]
        
        return score.data.cpu().numpy(), tag_seq
    
    def forward(self, input_var, tags, mask=None, usecuda=True):
        
        if mask is None:
            mask = Variable(torch.ones(*tags.size()).long())
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        
        maskedtags = tags.clone()
        maskedtags[mask==0] = -1
        
        features = features.view(-1, self.tagset_size)
        maskedtags = maskedtags.view(-1)
        loss = self.loss_function(features, maskedtags)
        
        return loss