import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from torch.nn import CrossEntropyLoss
import copy
from loss.MarginLoss import MarginLoss


class MylamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, args):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel(config)
        self.args = args
        self.num_labels = config.num_labels

        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.cross_entropy_loss_fct = CrossEntropyLoss()
        self.score_fun = args.score_function

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def _loss_fct(self, score, batch_y, regul):

        # logging.info("score: {}".format(score.shape))
        # logging.info("batch_y: {}".format(batch_y.shape))

        loss = self.cross_entropy_loss_fct(score.view(-1, self.num_labels), batch_y.view(-1))

        return loss

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask, labels=None
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=token_type_ids,
        )
        hidden_states = transformer_outputs[0]
        # logging.info("hidden_states: {}".format(hidden_states.shape))
        hidden_states = self.dropout(hidden_states)
        logits = self.score(hidden_states)

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        # logging.info("logits: {}".format(logits.shape))
        l2_reg = 0
        loss = self._loss_fct(score=pooled_logits, batch_y=labels, regul=l2_reg)
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=token_type_ids
        )
        hidden_states = transformer_outputs[0]
        pooled_output = self.dropout(hidden_states)
        logits = self.score(pooled_output)

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return pooled_logits
