import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss, NLLLoss
from loss.MarginLoss import MarginLoss

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


class MyBertForTokenHiddenState(BertPreTrainedModel):

    def __init__(self, config, args):
        super(MyBertForTokenHiddenState, self).__init__(config)

        logging.info("BERT MODEL.")

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier

        # self.classifier_tc = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier
        # self.classifier_lp = nn.Linear(config.hidden_size, self.num_labels)  # linear classifier

        self.softmax_func = nn.Softmax(dim=1)
        self.init_weights()

        self.score_fun = args.score_function
        self.args = args

        # self.gru = nn.GRU(config.hidden_size, args.gru_hidden_size, num_layers=args.gru_layers, bidirectional=True, batch_first=True)
        # self.classifier_gru = nn.Linear(args.gru_hidden_size*2, args.gru_hidden_size*2)
        # self.classifier_line_gru = nn.Linear(args.gru_hidden_size*2 + config.hidden_size, self.num_labels)

        self.cross_entropy_loss_fct = CrossEntropyLoss()
        # self.marginLoss = MarginLoss(adv_temperature=0.5, margin=args.margin)


    def _calc_gru(self,sequence_output):
        last_hidden_state = sequence_output
        # logging.info("last_hidden_state.shape: {0}".format(last_hidden_state.shape))
        #last_hidden_state.shape: [batch_size, sequence_length, hidden_size]
        h_n = None
        # for gru in self.gru:
        #     try:
        #         gru.flatten_parameters()
        #     except:
        #         pass
        #     output, h_n = gru(last_hidden_state)
        #     logging.info("h_n.shape: {0}".format(h_n.shape))
            #h_n.shape: [batch, num_layers*num_directions == 2, gru_hidden_size]    batch_size first
        output, h_n = self.gru(last_hidden_state)
        # logging.info("h_n.shape: {0}".format(h_n.shape)) # torch.Size([10, 16, 300])
        # logging.info("output.shape: {0}".format(output.shape)) # output.shape: torch.Size([16, 400, 600])
        output = output[:, -1, :]
        # output = h_n.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        output = self.classifier_gru(self.dropout(output))

        return output

    def _loss_fct(self, score, batch_y, regul):
        loss = self.cross_entropy_loss_fct(score.view(-1, self.num_labels), batch_y.view(-1))
        return loss

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.score_fun == "GRU_LINEAR_B_C":

            logging.info("GRU_LINEAR_B_C score function, c plus b, b with gru_linear.")

            # GRU layer (embedding for b)
            embedding_gre = self._calc_gru(sequence_output)

            # cat_embedding = torch.cat((pooled_output, embedding_gre), 1)
            # score = self.classifier_line_gru(self.dropout(cat_embedding))
             # linear for (c + b)
            cat_embedding = pooled_output + embedding_gre
            score = self.classifier(self.dropout(cat_embedding))

            # # regularization
            l2_reg = 0
            # for W in self.gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            # for W in self.classifier_gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            # for W in self.classifier_line_gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            #
            loss = self._loss_fct(score=score, batch_y=labels, regul=l2_reg)
        elif self.score_fun == "GRU_LINEAR_B":

            logging.info("GRU_LINEAR_B score function, only b, b with gru_linear.")

            # GRU layer (embedding for b)
            embedding_gre = self._calc_gru(sequence_output)
            score = self.classifier(self.dropout(embedding_gre))

            # # regularization
            l2_reg = 0
            # for W in self.gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            # for W in self.classifier_gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            # for W in self.classifier_line_gru.parameters():
            #     l2_reg = l2_reg + W.norm(2)
            #
            loss = self._loss_fct(score=score, batch_y=labels, regul=l2_reg)
        else:
            # logging.info("LINEAR score function, only c, c with linear..")
            # other score fun ..( Linear layer)
            pooled_output = self.dropout(pooled_output)

            score = self.classifier(pooled_output)
            # score = self.softmax_func(score) # not good result
            # # regularization
            l2_reg = 0
            # for W in self.classifier.parameters():
            #     l2_reg = l2_reg + W.norm(2)

            loss = self._loss_fct(score=score, batch_y=labels, regul=l2_reg)

        return loss
        # return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]


        # # score_function
        # if self.score_fun == "CNN":
        #     # Con layer
        #     score = self._calc_cnn(sequence_output)
        #
        # elif self.score_fun == "GRU":
        #     logging.info("GRU score function")
        #     score = self._calc_gru(sequence_output,input_ids)
        #
        if self.score_fun == "GRU_LINEAR_B_C":

            logging.info("GRU_LINEAR_B_C score function, c plus b, b with gru_linear.")

            # embedding_linear = self.classifier(self.dropout(pooled_output))
            # GRU layer
            # embedding_gre = self._calc_gru(sequence_output,input_ids)
            # cat_embedding = torch.cat((embedding_linear, embedding_gre), 1)
            # score = self.classifier_c_h(cat_embedding)

            embedding_gre = self._calc_gru(sequence_output)

            # cat_embedding = torch.cat((pooled_output, embedding_gre), 1)
            # score = self.classifier_line_gru(self.dropout(cat_embedding))

            cat_embedding = pooled_output + embedding_gre
            score = self.classifier(self.dropout(cat_embedding))

        elif self.score_fun == "GRU_LINEAR_B":

            logging.info("GRU_LINEAR_B score function, only b, b with gru_linear.")
            embedding_gre = self._calc_gru(sequence_output)
            score = self.classifier(self.dropout(embedding_gre))

        else:

            #Linear Layer
            # logging.info("LINEAR score function, only c, c with linear..")
            pooled_output = self.dropout(pooled_output)
            score = self.classifier(pooled_output)

            # logging.info("predict score: {0}".format(score))

        return score
