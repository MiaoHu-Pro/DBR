# define a class,ERDes,to finish the task of obtaining entity and relation description
# coding:utf-8
import json
import time
import pandas as pd
import torch
from torch.autograd import Variable
import os
import numpy as np
from utilities import entity_text_process, construct_entity_des
from utilities import Enti
from datetime import datetime

use_gpu = False


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obtain_entity_res(sub_x_obj, _entity_set):
    print("load title and description ...")

    all_entity_obj_list = []
    all_entity_description_list = []
    symbol = list(sub_x_obj[:, 1])

    # issue_with_type
    entity_issue_type = sub_x_obj[:, 2].tolist()
    entity_with_type = {}  # hadoop-321: "task"
    for i in range(len(symbol)):
        entity_with_type[symbol[i]] = entity_issue_type[i]

    for i in range(len(symbol)):
        entity_id = sub_x_obj[i, 0]
        entity_symbol = sub_x_obj[i, 1]

        entity_type = sub_x_obj[i, 2]
        entity_name = sub_x_obj[i, 3]
        entity_mention = sub_x_obj[i, 4]

        entity_des_word_list = construct_entity_des(entity_type, entity_name, entity_mention)

        entity = Enti(_id=entity_id, _symbol=entity_symbol,_type = entity_type,  _label=entity_name, _mention=entity_mention,
                      _entity_des_word_list=entity_des_word_list)

        all_entity_obj_list.append(entity)
        all_entity_description_list.append(entity_des_word_list)

    print("load title and description over ...")
    return all_entity_obj_list, all_entity_description_list


def read_entity2id(data_id_paht):
    f = open(data_id_paht)
    f.readline()

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()

    return np.array(x_obj)




def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """

    X = pd.read_csv(entity_obj_path, sep="\t", header=None, dtype=str)

    return np.array(X)


def read_data2id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s)
            data_id.append(id_list)
    data = np.array(data_id)
    return data


if torch.cuda.is_available():
    use_gpu = True


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            # _str = str(x.id) + '\t' + str(x.symbol) + '\t' + str(x.label) + '\t' + str(x.mention) + '\t' + str(
            #     x.neighbours) + '\n'

            _str = str(x.id) + '\t' + str(x.symbol) + '\t' + char.join(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_triple_descriptions(out_path, head_des, rel_des, tail_des):
    num_triples = len(head_des)
    ls = os.linesep
    head_len = []
    rel_len = []
    tail_len = []
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(num_triples):
            # print(i)
            head = head_des[i]
            rel = rel_des[i]
            tail = tail_des[i]
            head_len.append(len(head))
            rel_len.append(len(rel))
            tail_len.append(len(tail))

            _str = str(i) + '\t' + char.join(head) + '\t' + char.join(rel) + '\t' + char.join(tail) + '\n'
            #
            # _str = str(x.id) + '\t' + str(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')
    print("head len ", np.mean(head_len))
    print("rel len ", np.mean(rel_len))
    print("tail len ", np.mean(tail_len))


def read_initial_file(path):
    f = open(path)
    x_obj = {}

    i = 0
    for d in f:

        _issue = []
        d = d.strip()
        data = json.loads(d.replace('}{', '},{'))
        data['description'] = "no"

        # # syblom = data["bug_id"]
        # if i == 1000:
        #     print(data)

        x_obj[str(i)] = data

        i += 1
    f.close()
    return x_obj

def read_duplicate_bugs(path):

    f = open(path)
    f.readline()

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split(' ')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()
    test_set = np.array(x_obj)
    return test_set[:,0].tolist()

class ERDes(object):

    def __init__(self, _Paras):
        self.entity2Obj_path = _Paras['entity2Obj_path']
        self.entity2id_path = _Paras['entity2id_path']
        self.tokenizer = _Paras['tokenizer']
        self.initial_issue_path = _Paras['initial_issue_path']
        self.duplicateBug_path = _Paras['duplicateBug_path']

        self.positive_pair_path = _Paras['positive_pair_path']


        self.entity_res = None
        self.entity_symbol_set = []
        self.entity_index_set = []
        self.training_entity_index_set = []
        self.training_relation_index_set = []
        self.relation2id = []
        self.head_with_relations_and_tail_candidates = None
        self.tail_with_relations_and_head_candidates = None

        self.issue_obj = read_entity2obj(self.entity2Obj_path)

    def __len__(self):
        return len(self.id2symbol)

    def __contains__(self, bug):
        bugId = bug['bug_id'] if isinstance(bug, dict) else bug

        return bugId in self.bugById

    def get_entity_des(self):
        print("load entity object with textual information ...")
        symbol = list(self.issue_obj[:, 1])
        issue_time = list(self.issue_obj[:, -1])

        symbol2time_dic = {}
        for j in range(len(symbol)):
            symbol2time_dic[symbol[j]] = issue_time[j]

        # obtain_ entity id
        entity_id_read_file = read_entity2id(self.entity2id_path)
        self.entity_symbol_set = entity_id_read_file[:, 0].tolist()
        self.entity_index_set = entity_id_read_file[:, 1].tolist()

        id2symbol_dic = {}
        for i in range(len(self.entity_index_set)):
            id2symbol_dic[self.entity_index_set[i]] = self.entity_symbol_set[i]

        self.id2symbol = id2symbol_dic

        symbol2id_dic = {}
        for i in range(len(self.entity_index_set)):
            symbol2id_dic[self.entity_symbol_set[i]] = self.entity_index_set[i]

        self.symbol2id_dic = symbol2id_dic

        entityid_with_created_time = {} #
        for i in range(len(self.entity_index_set)):
            entityid_with_created_time[self.entity_index_set[i]] = symbol2time_dic[
                id2symbol_dic[self.entity_index_set[i]]]

        self.entityid_with_created_time = entityid_with_created_time
        all_entity_obj_list, all_entity_description_word_list = obtain_entity_res(self.issue_obj,
                                                                                  self.entity_symbol_set)
        self.entity_with_attributes = all_entity_obj_list
        self.entity_res = {'all_entity_obj_list': all_entity_obj_list,
                           'all_entity_description_word_list': all_entity_description_word_list}

        self._positive_pairs = read_entity2id(self.positive_pair_path)

        self.ent2tokens = {}
        for i in range(len(self.entity_with_attributes)):
            des = self.entity_with_attributes[i].entity_des
            self.ent2tokens[self.entity_with_attributes[i].id] = \
                self.tokenizer.tokenize(" ".join(des))
        self.rela2token = {}
        self.rela2token['1'] = self.tokenizer.tokenize("duplicate")
        self.rela2token['0'] = self.tokenizer.tokenize("not duplicate")


        print("end entity object with textual information ... \n")

        self.head_type2tail_type, self.entityid2Com =  self.get_headType2tailType()

        self.bugById = read_initial_file(self.initial_issue_path)

        self.duplicateIds  = read_duplicate_bugs(self.duplicateBug_path)
        format_pattern = "%Y-%m-%d %H:%M:%S"
        # Get oldest and newest duplicate bug report in dataset
        oldestDuplicateBug = (
            self.duplicateIds[0], entityid_with_created_time[self.duplicateIds[0]])

        for dupId in self.duplicateIds:

            creationDate = entityid_with_created_time[dupId]

            if datetime.strptime(oldestDuplicateBug[1], format_pattern) < datetime.strptime(creationDate, format_pattern):
                oldestDuplicateBug = (dupId, creationDate)

        self.candidates = []
        # for index, _date in entityid_with_created_time.items():
        #     self.candidates.append((index, _date))

        for index, _date in entityid_with_created_time.items():
            bugCreationDate = _date
            bugId = index

            # Remove bugs that their creation time is bigger than oldest duplicate bug
            if datetime.strptime(bugCreationDate, format_pattern) > datetime.strptime(oldestDuplicateBug[1], format_pattern) or (
                   datetime.strptime(bugCreationDate, format_pattern) == datetime.strptime(oldestDuplicateBug[1], format_pattern) and
                   int(bugId) > int(oldestDuplicateBug[0])):
                continue

            self.candidates.append((bugId, bugCreationDate))

        self.latestDateByMasterSetId = {}
        # Keep the timestamp of all reports in each master set
        for masterId, masterSet in self.getMasterSetById(map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for bugId in masterSet:
                bugCreationDate = entityid_with_created_time[bugId]

                ts_list.append((int(bugId), bugCreationDate))

            self.latestDateByMasterSetId[masterId] = ts_list

        # ID:{"id":AA, "duplicate":}
        self.masterIdByBugId = self.getMasterIdByBugId()
        self.masterSetById = self.getMasterSetById()

    def getMasterIdByBugId(self, bugs=None):
        masterIdByBugId = {}
        bugs = self.entity_index_set if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.bugById[bug]

            bug_symbol = bug['bug_id']
            dup_symbol = bug['dup_id']

            bugid = self.symbol2id_dic[bug_symbol]

            if len(dup_symbol) != 0:
                dupid = self.symbol2id_dic.get(dup_symbol)
                masterIdByBugId[bugid] = dupid
            else:
                masterIdByBugId[bugid] = bugid

        return masterIdByBugId
    #
    # def getMasterSetById(self):
    #
    #     symbol2time = {}
    #     for i in range(len(self.issue_obj)):
    #         symbol2time[self.issue_obj[i][1]] = self.issue_obj[i][-1]
    #
    #     re_order_train_pair = []
    #     for _data in list(self._positive_pairs):
    #         _pair = []
    #         head = int(_data[0])
    #         tail = int(_data[1])
    #         if head > tail:
    #             _pair.append(_data[0])
    #             _pair.append(_data[1])
    #         else:
    #             _pair.append(_data[1])
    #             _pair.append(_data[0])
    #         re_order_train_pair.append(_pair)
    #
    #     duplicate_cluster = {}
    #     for _each_train in re_order_train_pair:
    #         head = _each_train[0]
    #         tail = _each_train[1]
    #
    #         if head not in duplicate_cluster:
    #             duplicate_cluster[self.symbol2id_dic[head]] = [self.symbol2id_dic[tail]]
    #
    #         else:
    #             duplicate_cluster[self.symbol2id_dic[head]].append(self.symbol2id_dic[tail])
    #     masterSetById = duplicate_cluster
    #
    #     return masterSetById

    def getMasterSetById(self, bugs=None):

        masterSetById = {}

        bugs = self.entity_index_set

        for bug_id in bugs:

            bug = self.bugById[bug_id]

            dup_symbol = bug['dup_id']
            # print("dup_symbol", dup_symbol)

            if len(dup_symbol) != 0:
                dup_symbol = bug['dup_id'][0]
                dupid = self.symbol2id_dic.get(dup_symbol)
                masterSet = masterSetById.get(dupid, set())

                if len(masterSet) == 0:
                    masterSetById[dupid] = masterSet
                masterSet.add(self.symbol2id_dic[bug['bug_id']])

        # Insert id of the master bugs in your master sets
        for masterId, masterSet in masterSetById.items():
            if masterId in self:
                masterSet.add(masterId)

        return masterSetById

    def get_headType2tailType(self):

        _positive_pairs = self._positive_pairs

        symbol2Com = {}
        entityid2Com = {}
        for i in range(len(self.issue_obj)):
            symbol2Com[self.issue_obj[i][1]] = self.issue_obj[i][2]
            entityid2Com[self.issue_obj[i][0]] = self.issue_obj[i][2]

        train_pair_type = []
        for _data in list(_positive_pairs):
            _pair = []
            head = _data[0]
            tail = _data[1]

            type_head = symbol2Com[head]
            type_tail = symbol2Com[tail]

            _pair.append(type_head)
            _pair.append(type_tail)
            train_pair_type.append(_pair)

        head_type2tail_type = {}
        for _each_pair in train_pair_type:
            if _each_pair[0] not in head_type2tail_type:
                _pair = [_each_pair[1]]
                head_type2tail_type[_each_pair[0]] = _pair

            else:
                if _each_pair[1] not in head_type2tail_type[_each_pair[0]]:

                    head_type2tail_type[_each_pair[0]].append(_each_pair[1])

        return head_type2tail_type, entityid2Com


    def get_triple_des(self, _h, _r, _t):
        # print("get triple des begin ... ")

        all_entity_res_obj = self.entity_res['all_entity_obj_list']
        all_entity_des_word = self.entity_res['all_entity_description_word_list']

        head_index = _h
        tail_index = _t
        relation_index = _r

        head_obj = [all_entity_res_obj[i] for i in head_index]
        tail_obj = [all_entity_res_obj[i] for i in tail_index]

        head_description_list = [" ".join(all_entity_des_word[i]) for i in head_index]  # get head entity description

        tail_description_list = [" ".join(all_entity_des_word[i]) for i in tail_index]  # get tail entity

        relation_name = [self.relation2id[i][0] for i in relation_index]

        relation_description_word_list = []
        for i in range(len(relation_name)):
            rel_des = str(relation_name[i]) + ', ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
                i].label + ' .'

            relation_description_word_list.append(rel_des)

        return head_description_list, relation_description_word_list, tail_description_list

    def er_des_print(self):
        print(self.entity2id_path)


def obtain_train_triple_des(file_path, en_rel_des):
    print("obtain_train_triple_des ... \n")
    train_data_set_path = file_path + 'train_positive2id.txt'
    train = read_data2id(train_data_set_path)
    h = train[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = train[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = train[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'train_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_valid_triple_des(file_path, en_rel_des):
    print("obtain_valid_triple_des ... \n")
    valid_data_set_path = file_path + 'valid2id.txt'
    valid = read_data2id(valid_data_set_path)
    h = valid[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = valid[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = valid[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'valid_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_test_triple_des(file_path, en_rel_des):
    print("obtain_test_triple_des ... \n")
    test_data_set_path = file_path + 'test2id.txt'
    test = read_data2id(test_data_set_path)
    h = test[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = test[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = test[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]
    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)
    write_triple_descriptions(file_path + 'test_triple_des_4num_2step.txt', h_des, r_des, t_des)


if __name__ == "__main__":
    file_path = '../benchmarks/FB15K237/'
    Paras = {
        'num_neighbours': 4,
        'num_step': 2,
        'word_dim': 100,
        'all_triples_path': file_path + 'train.csv',
        'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
        'entity2id_path': file_path + 'entity2id.txt',
        'relation2id_path': file_path + 'relation2id.txt',
        'entity_des_path': file_path + 'entity2new_des_4nums_2step.txt',
    }
    en_rel_des = ERDes(_Paras=Paras)
    en_rel_des.get_entity_des()

    # train
    obtain_train_triple_des(file_path, en_rel_des)
    # valid
    obtain_valid_triple_des(file_path, en_rel_des)
    # test
    obtain_test_triple_des(file_path, en_rel_des)
