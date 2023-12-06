import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

from torch.nn import CrossEntropyLoss, NLLLoss
from loss.MarginLoss import MarginLoss
from typing import Optional, Tuple, Union

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()


class MyGPT2ForTokenClassificationCNN(GPT2PreTrainedModel):

    def __init__(self, config, args):
        GPT2PreTrainedModel.__init__(self, config)

        logging.info("GPT2CNN  MODEL.")
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        #
        logging.info("classifier_dropout : {0}".format(args.hidden_dropout_prob))
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.apply(self._init_weights)
        self.score_fun = args.score_function
        self.args = args

        self.cross_entropy_loss_fct = CrossEntropyLoss()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, (5, 5), bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, (5, 5), bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 5 * 1280, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim / 2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim / 2), 2, bias=False)

    def _cnn_encoder(self, x):
        # batch_size, sequence_max, hedden_size ==> 16*400*768
        b, r, c = x.shape
        # print(x.shape)
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
        print(x.shape)
        return x

    def _loss_fct(self, score, batch_y, regul):
        loss = self.cross_entropy_loss_fct(score.view(-1, self.num_labels), batch_y.view(-1))
        return loss

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None
    ):

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        batch_size, sequence_length = input_ids.shape[:2]

        # logging.info("LINEAR score function, only c, c with linear..")
        # hidden_states = transformer_outputs[0]
        # hidden_states = self.dropout(hidden_states)
        # logits = self.classifier(hidden_states)

        hidden_states = transformer_outputs[0]

        hidden_states = self.dropout(hidden_states)
        logits = self._cnn_encoder(hidden_states)
        logits = torch.squeeze(logits, dim=1)

        # logits = logits[torch.arange(batch_size, device=logits.device), -1]
        # print("logits.shape", logits.shape)
        loss = self._loss_fct(score=logits, batch_y=labels, regul=0)

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        logging.info("GPT2 predicting...")

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = transformer_outputs[0]

        batch_size, sequence_length = input_ids.shape[:2]


        # Linear Layer
        # sequence_output = self.dropout(sequence_output.clone().detach())
        # score = self.classifier(sequence_output)

        # cnn
        sequence_output = self.dropout(sequence_output.clone().detach())
        score = self._cnn_encoder(sequence_output)
        logits = torch.squeeze(score, dim=1)
        return score
