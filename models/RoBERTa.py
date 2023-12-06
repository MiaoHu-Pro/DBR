import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, NLLLoss
from loss.MarginLoss import MarginLoss

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()

class MyRobertaModel(RobertaPreTrainedModel):

    def __init__(self, config, args):
        super(MyRobertaModel, self).__init__(config)
        logging.info("RoBERTa MODEL.")
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # self.classifier_tc = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier
        # self.classifier_lp = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier

        self.softmax_func = nn.Softmax(dim=1)
        self.score_fun = args.score_function
        self.args = args

        self.loss_fct = CrossEntropyLoss()
        # self.marginLoss = MarginLoss(adv_temperature=0.5, margin=args.margin)

    def _calc_gru(self, sequence_output):
        last_hidden_state = sequence_output
        # logging.info("last_hidden_state.shape: {0}".format(last_hidden_state.shape))
        # last_hidden_state.shape: [batch_size, sequence_length, hidden_size]
        h_n = None
        # for gru in self.gru:
        #     try:
        #         gru.flatten_parameters()
        #     except:
        #         pass
        #     output, h_n = gru(last_hidden_state)
        #     logging.info("h_n.shape: {0}".format(h_n.shape))
        # h_n.shape: [batch, num_layers*num_directions == 2, gru_hidden_size]    batch_size first
        output, h_n = self.gru(last_hidden_state)
        # logging.info("h_n.shape: {0}".format(h_n.shape)) # torch.Size([10, 16, 300])
        # logging.info("output.shape: {0}".format(output.shape)) # output.shape: torch.Size([16, 400, 600])
        output = output[:, -1, :]
        # output = h_n.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        output = self.classifier_gru(self.dropout(output))

        return output

    def _loss_fct(self, score, batch_y, regul):

        reg_loss = self.loss_fct(score.view(-1, self.num_labels), batch_y.view(-1))
        logging.info("reg-loss : {0}".format(reg_loss))
        loss = reg_loss

        return loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None, labels=None):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # logging.info("LINEAR score function, only c, c with linear..")
        pooled_output = self.dropout(pooled_output)
        score = self.classifier(pooled_output)
        l2_reg = 0
        loss = self._loss_fct(score=score, batch_y=labels, regul=l2_reg)

        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Linear Layer
        # logging.info("LINEAR score function, only c, c with linear..")
        pooled_output = self.dropout(pooled_output)
        score = self.classifier(pooled_output)

        return score
