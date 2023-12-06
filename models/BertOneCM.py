import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss, NLLLoss
from loss.MarginLoss import MarginLoss

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


class MyBertForTokenHiddenStateOneClassModel(BertPreTrainedModel):

    def __init__(self, config, args):
        super(MyBertForTokenHiddenStateOneClassModel, self).__init__(config)

        logging.info("BERT with One Classification MODEL.")
        self.args = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier

        self.softmax_func = nn.Softmax(dim=1)
        self.init_weights()

        self.score_fun = args.score_function

        self.cross_entropy_loss_fct = CrossEntropyLoss()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, (5, 5), bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, (5, 5), bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 50 * 96, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim / 2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim / 2), 1, bias=False)

    def one_class_encoder(self, x):
        # batch_size, sequence_max, hedden_size ==> 16*400*768
        b, r, c = x.shape
        x = torch.reshape(x, (b, -1, r, c))
        # print("x.shape", x.shape)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # print("x.shape", x.shape)
        x = x.view(x.size(0), -1)
        # print("x.shape", x.shape)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        # print("x.shape", x.shape)
        return x

    def _loss_fct(self, score, batch_y, regul):
        reg_loss = self.cross_entropy_loss_fct(score.view(-1, self.num_labels), batch_y.view(-1))

        loss = reg_loss

        return loss

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # logging.info("LINEAR score function, only c, c with linear..")
        # pooled_output = self.dropout(pooled_output)
        # score = self.classifier(pooled_output)
        # loss = self._loss_fct(score=score, batch_y=labels, regul=0)

        # sequence_output --> [16, 400, 768]
        logits = self.one_class_encoder(sequence_output)
        logits = torch.squeeze(logits, dim=1)
        labels = labels.to(torch.float)
        labels = torch.squeeze(labels)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Linear Layer
        # logging.info("LINEAR score function, only c, c with linear..")
        # pooled_output = self.dropout(pooled_output)
        # score = self.classifier(pooled_output)

        logits = self.one_class_encoder(sequence_output)
        logits = torch.squeeze(logits, dim=1)
        sigmoid_logits = torch.sigmoid(logits)
        scores = logits

        return scores
