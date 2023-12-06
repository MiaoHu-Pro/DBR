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


class MyGPT2ForTokenClassification(GPT2PreTrainedModel):

    def __init__(self, config, args):
        GPT2PreTrainedModel.__init__(self, config)

        logging.info("GPT2 MODEL.")
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)

        # if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
        #     classifier_dropout = config.classifier_dropout
        # elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
        #     classifier_dropout = config.hidden_dropout
        # else:
        #     classifier_dropout = 0.1
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

        # pooled_output = transformer_outputs[1]

        logging.info("LINEAR score function ...")
        hidden_states = transformer_outputs[0] # 16*400*1024
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        logits = logits[torch.arange(batch_size, device=logits.device), -1]

        # # regularization
        # l2_reg = 0
        # for W in self.classifier.parameters():
        #     l2_reg = l2_reg + W.norm(2)

        loss = self._loss_fct(score=logits, batch_y=labels, regul=0)

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        logging.info("GPT2 predicting ... ")

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = transformer_outputs[0]
        batch_size, sequence_length = input_ids.shape[:2]

        # Linear Layer
        # logging.info("predicting LINEAR score function ...")
        sequence_output = self.dropout(sequence_output.clone().detach())
        score = self.classifier(sequence_output)
        score = score[torch.arange(batch_size), -1]

        return score
