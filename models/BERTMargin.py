import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel
from loss.MarginLoss import MarginLoss
import logging

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger = logging.getLogger(__name__)


class MyBertForTokenHiddenStateMargin(BertPreTrainedModel):

    def __init__(self, config, args):
        super(MyBertForTokenHiddenStateMargin, self).__init__(config)
        logging.info("BERT Margin MODEL.")

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.softmax_func = nn.Softmax(dim=1)
        self.apply(self.init_bert_weights)
        self.lmbda = args.lmbda
        self.margin = args.margin
        self.negative_times = args.negative_times
        self.marginLoss = MarginLoss(adv_temperature=0.5, margin=self.margin)
        self.criterion = nn.Softplus()
        self.max_seq_length = args.max_seq_length

    def _calc(self, input_ids, token_type_ids, attention_mask):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        score = self.classifier(pooled_output)

        return score

    def _get_positive_negative_scores(self, score, batch_y):

        positive_label = [i for i in range(len(batch_y)) if batch_y[i] == 1]
        negative_label = [i for i in range(len(batch_y)) if batch_y[i] == 0]

        positive_score = score[positive_label]
        positive_score = positive_score.view(-1, len(positive_score)).permute(1, 0)

        negative_score = score[negative_label]
        negative_score = negative_score.view(-1, self.negative_times).permute(0, 1)

        return positive_score, negative_score

    def _margin_loss(self, score, batch_y, regul):

        # p_score, n_score = self._get_positive_negative_scores(score, batch_y)
        # loss_res = self.marginLoss(p_score, n_score)

        margin_loss = self.margin_loss_fct(score[:, 0], score[:, 1])
        return margin_loss

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        score = self._calc(input_ids, token_type_ids, attention_mask)
        # regularization
        l2_reg = 0
        # for W in self.fc_layer1.parameters():
        #     l2_reg = l2_reg + W.norm(2)
        # for W in self.fc_layer2.parameters():
        #     l2_reg = l2_reg + W.norm(2)

        return self._margin_loss(score, labels, l2_reg)

    def predict(self, input_ids, token_type_ids, attention_mask):
        # logger.info("***** BERT_softPlus Running Prediction *****")
        score = self._calc(input_ids, token_type_ids, attention_mask)

        return score.cpu().data.numpy()
