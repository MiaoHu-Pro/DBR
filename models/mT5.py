import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mt5.modeling_mt5 import MT5Model, MT5PreTrainedModel, MT5Stack
from torch.nn import CrossEntropyLoss
import copy

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
torch.cuda.empty_cache()


class MyMT5ForClassification(MT5PreTrainedModel):
    def __init__(self, config, args):
        MT5PreTrainedModel.__init__(self, config)

        logging.info("MT5 MODEL.")
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.score_fun = args.score_function
        self.args = args

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = MT5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_value_states=None,
            use_cache=None,
            labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):

        batch_size, sequence_length = input_ids.shape[:2]

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        # the input of encoder and decoder are same which is input_ids
        decoder_input_ids = input_ids
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = logits[torch.arange(batch_size, device=logits.device), -1]

        loss = self._loss_fct(score=logits, batch_y=labels, regul=0)

        return loss

    def _loss_fct(self, score, batch_y, regul):

        reg_loss = self.loss_fct(score.view(-1, self.num_labels), batch_y.view(-1)) + self.args.lmbda * regul
        logging.info("reg-loss : {0}".format(reg_loss))
        loss = reg_loss

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        batch_size, sequence_length = input_ids.shape[:2]

        # Encode if needed (training, first prediction pass)

        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)

        hidden_states = encoder_outputs[0]

        # Decode
        # the input of encoder and decoder are same which is input_ids
        decoder_input_ids = input_ids
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        sequence_output = self.dropout(sequence_output)

        score = self.classifier(sequence_output)
        score = score[torch.arange(batch_size), -1]

        return score
