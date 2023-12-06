from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import glob
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from random import sample
from torch.nn import CrossEntropyLoss, NLLLoss
from loss.MarginLoss import MarginLoss
from sklearn import metrics
from data_process_utilities import KGProcessor, compute_metrics, EntDes
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer, BertConfig, RobertaConfig, RobertaTokenizer, GPT2Config, GPT2Tokenizer, \
    T5Config, T5Tokenizer, MT5Config, MT5Tokenizer, get_linear_schedule_with_warmup, LlamaTokenizer, LlamaConfig
from models.BERT import MyBertForTokenHiddenState
from models.RoBERTa import MyRobertaModel
from models.GPT2 import MyGPT2ForTokenClassification
from models.T5 import MyT5ForClassification
from models.mT5 import MyMT5ForClassification
from models.LLaMA import MylamaForSequenceClassification
from random import sample

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'bert': (BertConfig, MyBertForTokenHiddenState, BertTokenizer),
                 'roberta': (RobertaConfig, MyRobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, MyGPT2ForTokenClassification, GPT2Tokenizer),
                 't5': (T5Config, MyT5ForClassification, T5Tokenizer),
                 'llama': (LlamaConfig, MylamaForSequenceClassification, LlamaTokenizer),
                 'mt5': (MT5Config, MyMT5ForClassification, MT5Tokenizer), }


def train(args, processor, model, tokenizer, device):
    task_name = args.task_name.lower()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)  # num_labels = 2

    # read entities.txt, obtain entities
    entity_list = processor.get_entities(args.data_dir)

    # if args.local_rank == -1 or torch.distributed.get_rank() == 0:
    #     print("save EntDes ...")
    #     data_src = EntDes(_processor=processor, _label_list=label_list, _entity_list=entity_list,
    #                       _task_name=task_name)
    #     # save
    #     out_put = open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
    #         args.neighbours) + "_neighbours_data_src.pkl", 'wb')
    #     my_data_src = pickle.dumps(data_src)
    #     out_put.write(my_data_src)
    #     out_put.close()
    #
    # if not os.path.exists(args.pre_process_data) and args.local_rank in [-1, 0]:
    #     os.makedirs(args.pre_process_data)

    # get training data set
    train_examples = processor.get_train_examples(args.data_dir, args.negative)

    train_features = processor.to_convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    tr_loss = 0
    global_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    for e in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration No. " + str(e) + " epoch training",
                              disable=args.local_rank not in [-1, 0])
        model.train()
        tr_loss = 0

        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # epoch_iterator.set_description("loss {}".format(round(loss.item(), 5)))
            logger.info("epoch:{0} -- step:{1} -- loss:{2}".format(e, step, loss))

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     logs = {}
                #     if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args, model, tokenizer)
                #         for key, value in results.items():
                #             eval_key = 'eval_{}'.format(key)
                #             logs[eval_key] = value
                #
                #     loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                #     learning_rate_scalar = scheduler.get_lr()[0]
                #     logs['learning_rate'] = learning_rate_scalar
                #     logs['loss'] = loss_scalar
                #     logging_loss = tr_loss
                #
                #     for key, value in logs.items():
                #         tb_writer.add_scalar(key, value, global_step)
                #     print(json.dumps({**logs, **{'step': global_step}}))
                #
                # #save_steps , checkpoint
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # torch.distributed.get_rank() == 0 Save only once on process 0, avoiding saving duplicates
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)  #
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)  #
        torch.save(model_to_save.state_dict(), output_model_file)  # save model
        model_to_save.config.to_json_file(output_config_file)  # save config setting
        tokenizer.save_vocabulary(args.output_dir)  # save vocabulary

    return global_step, tr_loss / global_step


def gap_time(head_with_time, tail_with_time):
    format_pattern = "%Y-%m-%d %H:%M:%S"
    head_with_time = datetime.strptime(head_with_time, format_pattern)
    tail_with_time = datetime.strptime(tail_with_time, format_pattern)

    if tail_with_time > head_with_time:
        gap_hour = -1
        # print(tail_with_time, head_with_time)
    else:

        gap = (head_with_time - tail_with_time).total_seconds()
        gap_hour = round(gap / (24 * 3600), 3)

    return gap_hour


def write_true_ids_and_preds(args, _data_test, all_label_ids, preds):
    output_eval_file = os.path.join(args.output_dir, "triple_classification_label_result_ture_pre.txt")
    f = open(output_eval_file, 'a')
    length = len(all_label_ids)
    for i in range(length):
        head_id = _data_test[i].text_a
        r_id = _data_test[i].text_b
        tail_id = _data_test[i].text_c
        _str = str(head_id) + ' ' + str(tail_id) + ' ' + str(r_id) + '\t' + str(all_label_ids[i]) + '\t' + str(
            preds[i]) + '\n'
        f.write(_str)

    f.close()


from bug_dataset import BugDataset
from ranking import SunRanking, RecallRate, RankingResultFile
from bug_report_database import BugReportDatabase


def generateRecommendationList(anchorId, candidates, rankingScorer):
    args = rankingScorer["args"]
    model = rankingScorer["model"]
    tokenizer = rankingScorer["tokenizer"]
    processor = rankingScorer["processor"]
    label_list = rankingScorer["label_list"]

    device = rankingScorer["device"]

    entityid_with_created_time, id2symbol, symbol2id_dic, entityid2Com, head_type2tail_type = processor.get_triple_type_info()

    model.to(device)
    tail_corrupt_list = []
    for corrupt_ent in candidates:
        head = symbol2id_dic[str(anchorId)]
        tail = symbol2id_dic[str(corrupt_ent)]
        tmp_triple = [str(head), str(tail)]
        tail_corrupt_list.append(tmp_triple)

    eval_examples = processor.to_create_test_examples_for_duplicate_retrieval(tail_corrupt_list)
    eval_features = processor.to_convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    logger.info("***** Running Prediction *****")
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing\n"):
        model.eval()

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model.predict(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                   labels=None)
        if len(preds) == 0:
            batch_logits = logits.detach().cpu().numpy()
            preds.append(batch_logits)
        else:
            batch_logits = logits.detach().cpu().numpy()
            preds[0] = np.append(preds[0], batch_logits, axis=0)

    preds = preds[0]
    rel_values = preds[:, 0]
    # rel_values = torch.tensor(rel_values)
    similarityScores = rel_values

    # Remove pair (duplicateBug, duplicateBug) and create tuples with bug id and its similarity score.
    bugScores = [(bugId, score) for bugId, score in zip(candidates, similarityScores) if bugId != anchorId]
    # Sort  in descending order the bugs by probability of being duplicate
    sortedBySimilarity = sorted(bugScores, key=lambda x: x[1], reverse=False)

    return sortedBySimilarity


def logRankingResult(logger, rankingClass, rankingScorer, bugReportDatabase, rankingResultFilePath, epoch,
                     label=None, groupByMaster=True, recommendationListfn=generateRecommendationList):
    recallRateMetric = RecallRate(bugReportDatabase, groupByMaster=groupByMaster)
    rankingResultFile = RankingResultFile(rankingResultFilePath, bugReportDatabase) if rankingResultFilePath else None

    processor = rankingScorer["processor"]
    entityid_with_created_time, id2symbol, symbol2id_dic, entityid2Com, head_type2tail_type = processor.get_triple_type_info()

    filename, ext = os.path.splitext(rankingResultFilePath)
    file_prefix = "{}_{}{}".format(filename, int(random.randint(0, 10000000)), ext)

    positions = []
    pos_inf = []
    test_id = []

    top_tten_hit_count = 0
    top_ten_hit_count = 0
    top_three_hit_count = 0
    top_one_hit_count = 0

    hits = []
    for i in range(21):
        hits.append([])

    num_test = 0
    for i, duplicateBugId in enumerate(rankingClass.getDuplicateBugs()):
        logger.info('\n\nTest for No. {0} sample '.format(i))

        head_type = entityid2Com[symbol2id_dic[str(duplicateBugId)]]
        candidate_tail_type_distribution = head_type2tail_type.get(head_type)
        candidate_tail = []
        logger.info('duplicateBugId :{}'.format(duplicateBugId))
        candidates = rankingClass.getCandidateList(duplicateBugId)

        if i > 0 and i % 500 == 0:
            logger.info('RR calculation - {} duplicate reports were processed'.format(i))
        if len(candidates) == 0:
            if isinstance(rankingClass, SunRanking):
                # If the window of days is too small to contain a duplicate bug, so this can happen.
                logging.getLogger().warning("Bug {} has 0 candidates!".format(duplicateBugId))
            else:
                # This shouldn't happen with the other methodologies
                raise Exception("Bug {} has 0 candidates!".format(duplicateBugId))

            recommendation = candidates
        else:
            for _candidate in candidates:
                entity_type = entityid2Com[symbol2id_dic[str(_candidate)]]
                if entity_type in candidate_tail_type_distribution:
                    candidate_tail.append(_candidate)
            logger.info("candidates : {}".format(len(candidate_tail)))
            recommendation = recommendationListfn(duplicateBugId, candidate_tail, rankingScorer)

        # Update the metrics
        pos, correct_cand = recallRateMetric.update(duplicateBugId, recommendation)

        logger.info('pos :{}'.format(pos))

        logger.info("correct_cand : {}".format(correct_cand))

        f = open(file_prefix + '_tail_issue_ranks.txt', 'a')

        f.write(str(duplicateBugId) + "-->" + str(len(candidate_tail)) + "-->" + str(pos) + "-->" +
                str(correct_cand) + '\n')

        num_test += 1

        f.close()
        if pos != math.inf:

            positions.append(pos)

            logger.info('mean rank until now: {0}'.format(np.mean(positions)))
            if pos <= 1:
                top_one_hit_count += 1

            logger.info("hit@1 until now: {0}".format(top_one_hit_count * 1.0 / num_test))

            if pos <= 3:
                top_three_hit_count += 1

            logger.info("hit@3 until now: {0}".format(top_three_hit_count * 1.0 / num_test))

            if pos <= 10:
                top_ten_hit_count += 1
            logger.info("hit@10 until now: {0}".format(top_ten_hit_count * 1.0 / num_test))

            if pos <= 20:
                top_tten_hit_count += 1
            logger.info("hit@20 until now: {0}".format(top_tten_hit_count * 1.0 / num_test))

            f.write('Mean rank:' + "\t" + str(np.mean(positions)) + '\n')
            logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(positions))))

            # this could be done more elegantly, but here you go
            for hits_level in range(21):
                if pos <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        # if rankingResultFile:
        #     rankingResultFile.update(duplicateBugId, recommendation, pos, correct_cand)

        if pos == math.inf:
            pos_inf.append(1)
            logger.info('how many inf in pos_inf until now: {0}'.format(np.sum(pos_inf)))

    logger.info('how many inf in pos_inf: {0}'.format(np.sum(pos_inf)))
    # write the result to file:
    f = open(file_prefix + '_retrieval_results.txt', 'a')
    f.write(file_prefix + " :\n")

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        f.write('Hits @' + str(i + 1) + "\t" + str(np.mean(hits[i])) + '\n')

    logger.info('Mean rank: {0}'.format(np.mean(positions)))
    f.write('Mean rank:' + "\t" + str(np.mean(positions)) + '\n')

    # logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(hits[i]))))

    # f.write('Mean reciprocal rank right: ' + "\t" + str(np.mean(1. / np.array(hits[i]))) + '\n')
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(positions))))
    f.write('Mean reciprocal rank: ' + "\t" + str(np.mean(1. / np.array(positions))) + '\n')

    f.close()
    recallRateResult = recallRateMetric.compute()
    nDupBugs = len(rankingClass.getDuplicateBugs())

    # label = "_%s" % label if label and len(label) > 0 else ""

    # logger.info(
    # '[{}] Throughput: {} bugs per second (bugs={})'.format(label, (nDupBugs), nDupBugs,))

    for k, rate in recallRateResult.items():
        hit = recallRateMetric.hitsPerK[k]
        total = recallRateMetric.nDuplicate
        logger.info(
            {'type': "metric", 'label': 'recall_rate%s' % (label), 'k': k, 'rate': rate, 'hit': hit,
             'total': total,
             'epoch': epoch})

        logger.info('recall_rate%s_%d' % (label, k), rate)

    valid_queries = np.asarray(positions)
    MAP_sum = (1 / valid_queries).sum()
    MAP = MAP_sum / valid_queries.shape[0]
    logger.info('MAP: {}'.format(MAP))

    logger.info(
        {'type': "metric", 'label': 'MAP', 'value': MAP, 'sum': MAP_sum, 'total': valid_queries.shape[0],
         'epoch': epoch})

    logger.info('{}'.format(positions))


def do_duplicate_bug_retrieval(args, model, tokenizer, processor, label_list, device):
    args.data_dir + '/' + "test_" + args.data_dir[7:].strip() + ".txt"

    recallRateOpt = {
        "type": "sun2011",
        "dataset": args.data_dir + '/' + "test_" + args.data_dir[7:].strip() + ".txt",
        "result_file": str(args.output_dir) + str(args.data_dir[7:]) + "_batch_size_" + str(
            args.per_gpu_train_batch_size) + "_lr_" + str(
            args.learning_rate) + "_max_seq_" + str(args.max_seq_length) + "_epochs_" + str(
            args.num_train_epochs) + "_negative_" + str(
            args.negative) + "_gap_time_" + str(args.gap_time) + "_test_file_" + str(args.test_file)[:-4],
        "group_by_master": True,
        "window": 365
    }
    rankingScorer = {
        "args": args,
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "label_list": label_list,
        "device": device
    }

    bugReportDatabase = BugReportDatabase.fromJson(
        args.data_dir + '/' + args.data_dir[7:].strip() + "_initial.json", )  # eclipse_initial.json

    rankingClass = None
    group_by_master = None

    # Calculate recall rate
    recommendation_fn = generateRecommendationList
    if recallRateOpt['type'] != 'none':
        if recallRateOpt['type'] == 'sun2011':
            logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
            recallRateDataset = BugDataset(recallRateOpt['dataset'])

            rankingClass = SunRanking(bugReportDatabase, recallRateDataset, recallRateOpt['window'])
            # We always group all bug reports by master in the results in the sun 2011 methodology
            group_by_master = True

        logRankingResult(logger, rankingClass, rankingScorer, bugReportDatabase,
                         recallRateOpt["result_file"], 0, None, group_by_master, recommendationListfn=recommendation_fn)

    #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")

    parser.add_argument("--pre_process_data",
                        default=None,
                        type=str,
                        required=True,
                        help="the path of pre_process training data .")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=400,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_duplicate_bug_retrieval",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--do_duplicate_issues_classification",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--load_pre_data",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--random_candidate_tails",
                        default=3000,
                        type=int,
                        help="the number of candidates randomly.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--score_function', type=str, default='linear', help='choose a score function, linear or CNN')
    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                        help='choose a loss function, cross_entropy or  max_margin')
    parser.add_argument('--negative',
                        type=int,
                        default=3,
                        help="how many negative entities")

    parser.add_argument('--neighbours',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help="random seed for initialization")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument('--gap_time',
                        type=float,
                        default=60,
                        help="the default is set into 2 months.")

    parser.add_argument('--test_file',
                        type=str,
                        default="test_positive2id.txt",
                        help="choose different test file.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--threshold_value", default=1, type=float,
                        help="")
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float,
                        help="")
    args = parser.parse_args()
    logger.info(" gap_time is : {0}.".format(args.gap_time))
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()

    else:
        # DDP
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # set seed
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.load_pre_data:

        print("read Ent src ... ")
        with open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
                args.neighbours) + "_neighbours_data_src.pkl", 'rb') as file:
            data_src = pickle.loads(file.read())
        # load data
        processor = data_src.processor
        label_list = data_src.label_list
        entity_list = data_src.entity_list
        task_name = data_src.task_name
        num_labels = len(label_list)

    else:
        processors = {"kg": KGProcessor, }
        task_name = args.task_name.lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)
        processor = processors[task_name](args=args)
        label_list = processor.get_labels(args.data_dir)
        num_labels = len(label_list)  # num_labels = 2

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        args=args)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    parameters_to_file = str(args.output_dir) + 'parameters_for_model.txt'
    f = open(parameters_to_file, 'a')

    logger.info("Model config %s", str(config))
    f.write("\nModel config Parameters following :\n")
    logger.info("==== Model config Parameters: =====")
    for attr, value in sorted(config.__dict__.items()):
        logger.info('\t{}={}'.format(attr, value))
        f.write(attr + "\t" + str(value) + '\n')
    logger.info("==== config Parameters End =====\n")
    f.write("\n==== config Parameters End ===== :\n")

    f.write("\n\nTraining/Evaluation Parameters following :\n")
    logger.info("==== Training/Evaluation Parameters: =====")
    for attr, value in sorted(args.__dict__.items()):
        logger.info('\t{}={}'.format(attr, value))
        f.write(attr + "\t" + str(value) + '\n')
    logger.info("==== Parameters End =====\n")
    f.write("\n==== Training/Evaluation Parameters End ===== :\n")
    f.close()

    # create new des for entity
    processor.get_entity_res(file_path=args.data_dir, tokenizer=tokenizer)

    logger.info("====  Data  Preparation End  =====\n")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    if args.do_train:
        global_step, tr_loss = train(args, processor, model, tokenizer, device)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # do_duplicate_bug_retrieval
    if args.do_duplicate_bug_retrieval and args.local_rank in [-1, 0]:
        logger.info("+++++++++++do_duplicate_bug_retrieval+++++++++++++++=")
        logger.info("load config and fine-tuned model... !")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir, args=args)
        logger.info("to run do_duplicate_bug_retrieval!")
        do_duplicate_bug_retrieval(args, model, tokenizer, processor, label_list, device)


if __name__ == "__main__":
    main()

# python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/red_hat   --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_RedHat_data_77K_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt  --do_train
