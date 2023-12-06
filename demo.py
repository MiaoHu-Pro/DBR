from datetime import datetime
import math
import os
import pickle
from functools import partial
from multiprocessing import Pool
import re

import numpy as np
import random

import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def test():
    s = np.random.randint(low=0, high=10, size=5)
    list1 = [i for i in range(10)]
    num1 = random.sample(list1, 8)
    print(num1)


def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_obj_path)

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
    X = np.array(x_obj)

    return X


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep
    leng = len(all_data)
    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for j in range(leng):
            #
            _str = str(j) + '\t' + all_data[j][0] + '\t' + all_data[j][1] + '\t' + all_data[j][2] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_file():
    s = "Miao Hu"
    l = ["A", "B", "C", "D"]
    str = ' '
    f = open("k.txt", "w")
    f.write(s + "\t" + str.join(l))
    f.close()


def test2():
    word_list = [['I', 'am', 'a', 'student'], ['I', 'am', 'a', 'teacher'], ['I', 'am', 'a', 'docter'], ]
    char = " "

    new_str = [" ".join(word_list[i]) for i in range(len(word_list))]

    print(new_str)


def test():
    i = ['1', '1', '1', '1', '1', '1']
    print([int(i) for i in i])

    labels = [1, 2, 3, 4, 5]
    labs = torch.tensor(1, dtype=torch.long)
    print(labs)
    labs = str(labs.cpu().detach().numpy().tolist())
    print(type(labs))


def test2():
    aList = [123, 'a', 'b', '.', 'c', '.', 'd', 'd', 'd', '.', 'd', 'd']

    B = ['am', '##eric', '##an', 'pie', ',', 'that', 'is', 'us', 'comedy', 'film', '.',
         'am', '##eric', '##an', 'pie', 'has', 'a', 'relationship', 'of', 'award', 'award',
         'winning', 'work', 'awards', 'won', 'award', 'award', 'honor', 'award', 'winner', 'with', 'ta', '##ra', 're',
         '##id', '.',
         'am', '##eric', '##an', 'pie', 'has', 'a', 'relationship', 'of', 'film', 'film', 'release', 'date', 's',
         'film',
         'film', 'regional', 'release', 'date', 'film', 'release', 'distribution', 'medium', 'with', 'television', '.',
         'am', '##eric', '##an', 'pie', 'has', 'a', 'relationship', 'of', 'film', 'film', 'release', 'date', 's',
         'film', 'film',
         'regional', 'release', 'date', 'film', 'release', 'region', 'with', 'nor', '##way', '.', 'am', '##eric',
         '##an', 'pie',
         'has', 'a', 'relationship', 'of', 'film', 'film', 'release', 'date', 's', 'film', 'film', 'regional', 'release'
        , 'date', 'film', 'release', 'region', 'with', 's', '##lov', '##aki', '##a', '.',

         'am', '##eric', '##an', 'pie',
         'has', 'a', 'relationship', 'of', 'film', 'film', 'language', 'with', 'en', '##gli', '##sh', 'language',

         '.', 'am', '##eric', '##an', 'pie', 'has', 'a', 'relationship', 'of', 'film', 'film', 'release', 'date',
         's', 'film', 'film', 'regional', 'release', 'date', 'film', 'release', 'region', 'with', 'p', '##ola', '##nd',
         '.']

    # B = ['am', '##eric', '##an', 'pie', ',', 'that', 'is', 'us', 'comedy', 'film', '.',]

    print(len(B))
    index_d = [i for i in range(len(B)) if B[i] == '.']
    print(index_d)
    del B[index_d[-2]:-1]

    print(B)
    print(len(B))

    index_d = [i for i in range(len(B)) if B[i] == '.']
    del B[index_d[-2]:-1]

    print(B)
    print(len(B))
    print(int(20.7))

    # aList.pop()

    # print("List : ", aList)


"""

failed to deal with  
[125, 146] ['gym', '##nos', '##per', '##mo', '##phy', '##ta', ',', 'that', 'is', 'plants', 'having', 'naked', 
'seeds', 'not', 'enclosed', 'in', 'an', 'o', '##vary', 'in', 'some', 'systems', 'considered', 'a', 'class', 'gym', 
'##nos', '##per', '##ma', '##e', 'and', 'in', 'others', 'a', 'division', 'gym', '##nos', '##per', '##mo', '##phy', 
'##ta', 'comprises', 'three', 'subdivisions', 'or', 'classes', 'c', '##y', '##ca', '##do', '##phy', '##tina', 
'class', 'c', '##y', '##ca', '##do', '##ps', '##ida', 'and', 'g', '##net', '##op', '##hy', '##tina', 'class', 
'g', '##net', '##ops', '##ida', 'and', 'con', '##ifer', '##op', '##hy', '##tina', 'class', 'con', '##ifer', '##ops', 
'##ida', 'in', 'some', 'classification', '##s', 'the', 'con', '##ifer', '##op', '##hy', '##tina', 'are', 'divided',
 'into', 'three', 'groups', 'pin', '##op', '##hy', '##tina', 'class', 'pin', '##ops', '##ida', 'and', 'g', '##ink', 
 '##go', '##phy', '##tina', 'class', 'g', '##ink', '##go', '##ps', '##ida', 'and', 'tax', '##op', '##hy', '##tina',
  'class', 'tax', '##ops', '##ida', 
  
  '.', 'gym', '##nos', '##per', '##mo', '##phy', '##ta', 'has', 'a', 
  'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'gym', '##nos', '##per', '##mous', 'tree', '.'] 
  
[21] ['member', 'me', '##ron', '##ym', 'which', 'is', 'between', 'gym', '##nos', '##per', '##mo', '##phy', 
'##ta', 'and', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', '.']
 
 [57, 72, 87, 101, 119, 131] ['subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', ',', 'that', 'is', 'palm', '##like', 'gym', '##nos', '##per', '##ms', 'includes', 'the', 'surviving', 'order', 'c', '##y', '##ca', '##dale', '##s', 'and', 'several', 'extinct', 'orders', 'possibly', 'not', 'a', 'natural', 'group', 'in', 'some', 'systems', 'considered', 'a', 'class', 'c', '##y', '##ca', '##do', '##ps', '##ida', 'and', 'in', 'others', 'a', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'or', 'c', '##y', '##ca', '##do', '##phy', '##ta', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'h', '##yper', '##ny', '##m', 'with', 'class', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'order', 'ben', '##nett', '##ital', '##es', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'order', 'c', '##y', '##ca', '##dale', '##s', '.']
failed to deal with  [125, 146] ['gym', '##nos', '##per', '##mo', '##phy', '##ta', ',', 'that', 'is', 'plants', 'having', 'naked', 'seeds', 'not', 'enclosed', 'in', 'an', 'o', '##vary', 'in', 'some', 'systems', 'considered', 'a', 'class', 'gym', '##nos', '##per', '##ma', '##e', 'and', 'in', 'others', 'a', 'division', 'gym', '##nos', '##per', '##mo', '##phy', '##ta', 'comprises', 'three', 'subdivisions', 'or', 'classes', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'class', 'c', '##y', '##ca', '##do', '##ps', '##ida', 'and', 'g', '##net', '##op', '##hy', '##tina', 'class', 'g', '##net', '##ops', '##ida', 'and', 'con', '##ifer', '##op', '##hy', '##tina', 'class', 'con', '##ifer', '##ops', '##ida', 'in', 'some', 'classification', '##s', 'the', 'con', '##ifer', '##op', '##hy', '##tina', 'are', 'divided', 'into', 'three', 'groups', 'pin', '##op', '##hy', '##tina', 'class', 'pin', '##ops', '##ida', 'and', 'g', '##ink', '##go', '##phy', '##tina', 'class', 'g', '##ink', '##go', '##ps', '##ida', 'and', 'tax', '##op', '##hy', '##tina', 'class', 'tax', '##ops', '##ida', '.', 'gym', '##nos', '##per', '##mo', '##phy', '##ta', 'has', 'a', 'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'gym', '##nos', '##per', '##mous', 'tree', '.'] [21] ['member', 'me', '##ron', '##ym', 'which', 'is', 'between', 'gym', '##nos', '##per', '##mo', '##phy', '##ta', 'and', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', '.'] [57, 72, 87, 101, 119, 131] ['subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', ',', 'that', 'is', 'palm', '##like', 'gym', '##nos', '##per', '##ms', 'includes', 'the', 'surviving', 'order', 'c', '##y', '##ca', '##dale', '##s', 'and', 'several', 'extinct', 'orders', 'possibly', 'not', 'a', 'natural', 'group', 'in', 'some', 'systems', 'considered', 'a', 'class', 'c', '##y', '##ca', '##do', '##ps', '##ida', 'and', 'in', 'others', 'a', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'or', 'c', '##y', '##ca', '##do', '##phy', '##ta', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'h', '##yper', '##ny', '##m', 'with', 'class', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'order', 'ben', '##nett', '##ital', '##es', '.', 'subdivision', 'c', '##y', '##ca', '##do', '##phy', '##tina', 'has', 'a', 'relationship', 'of', 'member', 'me', '##ron', '##ym', 'with', 'order', 'c', '##y', '##ca', '##dale', '##s', '.']

"""


def test3():
    from torch import nn

    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    print(input)

    target = torch.empty(3, dtype=torch.long).random_(5)
    print(target)

    output = loss(input, target)
    print(output)

    output.backward()


def test4():
    label_list = [str(i) for i in range(237)]
    num_labels = len(label_list)

    print(label_list)


def get_rank(batch_test, i):
    batch = batch_test[i]

    rank1 = batch + 1
    rank2 = batch + 2

    return [rank1.cpu().numpy().tolist(), rank2.cpu().numpy().tolist()]


def pool_test():
    test_triples = [i for i in range(1000)]

    triple_data = TensorDataset(torch.tensor(test_triples))

    triple_dataloader = DataLoader(triple_data, batch_size=10)
    all_rank = []
    all_rank1 = []
    all_rank2 = []
    for step, batch_test in enumerate(tqdm(triple_dataloader, desc="\n batch test")):
        pool = Pool(processes=4)

        pfunc = partial(get_rank, batch_test)
        index_list1 = [i for i in range(len(batch_test))]

        temp_all_rank = pool.map(pfunc, index_list1)

        # print(temp_all_rank)

        # all_rank += temp_all_rank

        all_rank1 += temp_all_rank[0][0]
        all_rank2 += temp_all_rank[0][1]

        pool.close()
        pool.join()

    # print(all_rank)

    # all_rank = np.array(all_rank1)
    # all_rank1 = all_rank[:, 0]
    # all_rank2 = all_rank[:, 1]

    print("all_rank1 : ", all_rank1)
    # print("all_rank2 : ", all_rank2)


class Student():
    def __init__(self, _id, _name, _age):
        self.id = _id
        self.name = _name
        self.age = _age


class Teacher():
    def __init__(self, _id, _name, _age):
        self.id = _id
        self.name = _name
        self.age = _age


class Example():
    def __init__(self, student_list, teacher_list):
        self.student_list = student_list
        self.teacher_list = teacher_list


def save_obj():
    student_list = []

    for i in range(10):
        _obj = Student(i + 1, "Tom_" + str(i), i + 21)

        student_list.append(_obj)

    teacher_list = []
    for i in range(10):
        _obj = Teacher(i + 100, "Jack_" + str(i), i + 43)

        teacher_list.append(_obj)

    my_example = Example(student_list, teacher_list)

    # 保存
    out_put = open("./src/my_example.pkl", 'wb')

    my_example = pickle.dumps(my_example)
    out_put.write(my_example)
    out_put.close()

    # 读取kdtree 缩短时间

    with open("./src/my_example.pkl", 'rb') as file:
        my_example = pickle.loads(file.read())

    print("打印类对象 。。 ")
    length = len(my_example.student_list)
    for i in range(length):
        _str = str(my_example.student_list[i].id) + " " + my_example.student_list[i].name + " " + str(
            my_example.student_list[i].age)
        _str_2 = str(my_example.teacher_list[i].id) + " " + my_example.teacher_list[i].name + " " + str(
            my_example.teacher_list[i].age)

        print(_str, "\n", _str_2)
        print(" ------ ")


import cytoolz as ct
from gensim.parsing import preprocessing


def clean_wordclouds(line, stage=4):
    function_words_single = ["the", "of", "and", "to", "a", "in", "i", "he", "that", "was", "it", "his", "you", "with",
                             "as", "for", "had", "is", "her", "not", "but", "at", "on", "she", "be", "have", "by",
                             "which", "him", "they", "this", "from", "all", "were", "my", "we", "one", "so", "said",
                             "me", "there", "or", "an", "are", "no", "would", "their", "if", "been", "when", "do",
                             "who", "what", "them", "will", "out", "up", "then", "more", "could", "into", "man", "now",
                             "some", "your", "very", "did", "has", "about", "time", "can", "little", "than", "only",
                             "upon", "its", "any", "other", "see", "our", "before", "two", "know", "over", "after",
                             "down", "made", "should", "these", "must", "such", "much", "us", "old", "how", "come",
                             "here", "never", "may", "first", "where", "go", "s", "came", "men", "way", "back",
                             "himself", "own", "again", "say", "day", "long", "even", "too", "think", "might", "most",
                             "through", "those", "am", "just", "make", "while", "went", "away", "still", "every",
                             "without", "many", "being", "take", "last", "shall", "yet", "though", "nothing", "get",
                             "once", "under", "same", "off", "another", "let", "tell", "why", "left", "ever", "saw",
                             "look", "seemed", "against", "always", "going", "few", "got", "something", "between",
                             "sir", "thing", "also", "because", "yes", "each", "oh", "quite", "both", "almost", "soon",
                             "however", "having", "t", "whom", "does", "among", "perhaps", "until", "began", "rather",
                             "herself", "next", "since", "anything", "myself", "nor", "indeed", "whose", "thus",
                             "along", "others", "till", "near", "certain", "behind", "during", "alone", "already",
                             "above", "often", "really", "within", "used", "use", "itself", "whether", "around",
                             "second", "across", "either", "towards", "became", "therefore", "able", "sometimes",
                             "later", "else", "seems", "ten", "thousand", "don", "certainly", "ought", "beyond",
                             "toward", "nearly", "although", "past", "seem", "mr", "mrs", "dr", "thou", "except",
                             "none", "probably", "neither", "saying", "ago", "ye", "yourself", "getting", "below",
                             "quickly", "beside", "besides", "especially", "thy", "thee", "d", "unless", "three",
                             "four", "five", "six", "seven", "eight", "nine", "hundred", "million", "billion", "third",
                             "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "amp", "m", "re", "u",
                             "via", "ve", "ll", "th", "lol", "pm", "things", "w", "didn", "doing", "doesn", "r", "gt",
                             "n", "st", "lot", "y", "im", "k", "isn", "ur", "hey", "yeah", "using", "vs", "dont", "ok",
                             "v", "goes", "gone", "lmao", "happen", "wasn", "gotta", "nd", "okay", "aren", "wouldn",
                             "couldn", "cannot", "omg", "non", "inside", "iv", "de", "anymore", "happening",
                             "including", "shouldn", "yours", ]

    if stage > 3:
        # Remove links, hashtags, at-mentions, mark-up, and "RT"
        line = re.sub(r"http\S+", "", line)
        line = re.sub(r"@\S+", "", line)
        line = re.sub(r"#\S+", "", line)
        line = re.sub("<[^>]*>", "", line)
        line = line.replace(" RT", "").replace("RT ", "")

    if stage > 2:
        # Remove punctuation and extra spaces
        line = ct.pipe(line,
                       preprocessing.strip_tags,
                       preprocessing.strip_punctuation,
                       preprocessing.strip_numeric,
                       preprocessing.strip_non_alphanum,
                       preprocessing.strip_multiple_whitespaces
                       )

    if stage > 1:
        # Strip and lowercase
        line = line.lower().strip().lstrip().split()
    else:
        line = line.split()

    if stage > 0:
        line = [x for x in line if x not in function_words_single]

    return line


def clean(line):
    function_words_single = ["the", "of", "and", "to", "a", "in", "i", "he", "that", "was", "it", "his", "you", "with",
                             "as", "for", "had", "is", "her", "not", "but", "at", "on", "she", "be", "have", "by",
                             "which", "him", "they", "this", "from", "all", "were", "my", "we", "one", "so", "said",
                             "me", "there", "or", "an", "are", "no", "would", "their", "if", "been", "when", "do",
                             "who", "what", "them", "will", "out", "up", "then", "more", "could", "into", "man", "now",
                             "some", "your", "very", "did", "has", "about", "time", "can", "little", "than", "only",
                             "upon", "its", "any", "other", "see", "our", "before", "two", "know", "over", "after",
                             "down", "made", "should", "these", "must", "such", "much", "us", "old", "how", "come",
                             "here", "never", "may", "first", "where", "go", "s", "came", "men", "way", "back",
                             "himself", "own", "again", "say", "day", "long", "even", "too", "think", "might", "most",
                             "through", "those", "am", "just", "make", "while", "went", "away", "still", "every",
                             "without", "many", "being", "take", "last", "shall", "yet", "though", "nothing", "get",
                             "once", "under", "same", "off", "another", "let", "tell", "why", "left", "ever", "saw",
                             "look", "seemed", "against", "always", "going", "few", "got", "something", "between",
                             "sir", "thing", "also", "because", "yes", "each", "oh", "quite", "both", "almost", "soon",
                             "however", "having", "t", "whom", "does", "among", "perhaps", "until", "began", "rather",
                             "herself", "next", "since", "anything", "myself", "nor", "indeed", "whose", "thus",
                             "along", "others", "till", "near", "certain", "behind", "during", "alone", "already",
                             "above", "often", "really", "within", "used", "use", "itself", "whether", "around",
                             "second", "across", "either", "towards", "became", "therefore", "able", "sometimes",
                             "later", "else", "seems", "ten", "thousand", "don", "certainly", "ought", "beyond",
                             "toward", "nearly", "although", "past", "seem", "mr", "mrs", "dr", "thou", "except",
                             "none", "probably", "neither", "saying", "ago", "ye", "yourself", "getting", "below",
                             "quickly", "beside", "besides", "especially", "thy", "thee", "d", "unless", "three",
                             "four", "five", "six", "seven", "eight", "nine", "hundred", "million", "billion", "third",
                             "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "amp", "m", "re", "u",
                             "via", "ve", "ll", "th", "lol", "pm", "things", "w", "didn", "doing", "doesn", "r", "gt",
                             "n", "st", "lot", "y", "im", "k", "isn", "ur", "hey", "yeah", "using", "vs", "dont", "ok",
                             "v", "goes", "gone", "lmao", "happen", "wasn", "gotta", "nd", "okay", "aren", "wouldn",
                             "couldn", "cannot", "omg", "non", "inside", "iv", "de", "anymore", "happening",
                             "including", "shouldn", "yours", ]
    # function_words_single =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    # Remove links, hashtags, at-mentions, mark-up, and "RT"
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"@\S+", "", line)
    line = re.sub(r"#\S+", "", line)
    line = re.sub("<[^>]*>", "", line)
    line = line.replace(" RT", "").replace("RT ", "")

    # Remove punctuation and extra spaces
    line = ct.pipe(line,
                   preprocessing.strip_tags,
                   preprocessing.strip_punctuation,
                   preprocessing.strip_numeric,
                   preprocessing.strip_non_alphanum,
                   preprocessing.strip_multiple_whitespaces
                   )

    # Strip and lowercase
    line = line.lower().strip().lstrip().split()

    line = [x for x in line if x not in function_words_single]

    return line


import os


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pkl':
                L.append(file)
    return L


def test_sigmod():

    import torch
    import torch.nn as nn


    # s = torch.tensor([[-0.5, -0.7, 0.5, 0.7,0.5, 0.7]])
    # liner_layer = torch.nn.Linear(6, 2)
    # x_input = liner_layer(s)
    # print(x_input)
    # # 计算输入softmax，此时可以看到每一行加到一起结果都是1
    # softmax_func = nn.Softmax(dim=1)
    # soft_output = softmax_func(x_input)
    # print('soft_output:\n', soft_output)
    # soft_output = softmax_func(soft_output)
    # print('soft_output:\n', soft_output)


    # r_a = torch.sigmoid(liner_layer(s))
    # print(r_a)

    x_input = torch.randn(2, 2)  # 随机生成输入
    print('x_input:\n', x_input)
    y_target = torch.tensor([1,0])  # 设置输出具体值 print('y_target\n',y_target)

    # 计算输入softmax，此时可以看到每一行加到一起结果都是1
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(x_input)
    print('soft_output:\n', soft_output)

    # soft_output = softmax_func(x_input)
    # print('soft_output:\n', soft_output)


    # 在softmax的基础上取log
    log_output = torch.log(soft_output)
    print('log_output:\n', log_output)

    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func = nn.NLLLoss()
    nlloss_output = nllloss_func(torch.log(soft_output), y_target)
    print('nlloss_output:\n', nlloss_output)


    # 对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    # logsoftmax_func = nn.LogSoftmax(dim=1)
    # logsoftmax_output = logsoftmax_func(x_input)
    # print('logsoftmax_output:\n', logsoftmax_output)
    #
    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    # nllloss_func = nn.NLLLoss()
    # nlloss_output = nllloss_func(logsoftmax_output, y_target)
    # print('nlloss_output:\n', nlloss_output)




    # 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(x_input, y_target)

    print('crossentropyloss_output:\n', crossentropyloss_output)


"""
CrossEntropyLoss(x_input, y_target)  



soft_output = softmax_func(x_input) -- >  log_output = torch.log(soft_output) ->  nllloss_func(log_output, y_target)

soft_output = softmax_func(x_input) + log_output = torch.log(soft_output) = LogSoftmax



x_input = [a1,a2]

softmax_func(x_input) = [p1, p2] (p1 + p2 = 1)


CrossEntropyLoss(x_input, y_target)

nllloss_func(torch.log(softmax_func(x_input)), y_target)

"""

        
def _read_tsv(input_file):
    """Reads a tab separated value file.

    修改：
    reading train.csv will be changed with reading train_positive2id.txt, obtain index
    """
    data = pd.read_csv(input_file)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s.strip())
            data_id.append(id_list)

    return data_id

def _read_triple_tsv(input_file):
    """Reads a tab separated value file.

    修改：
    reading train.csv will be changed with reading train_positive2id.txt, obtain index
    """
    data = pd.read_csv(input_file)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split('\t')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s.strip())
            data_id.append(id_list)

    return data_id

def write_triples_2_id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + ' ' + str(data[k][1]) + ' ' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')

def combine_m_f_r():
    m_f_path = "./data/WN18RR/test2id_m_f.txt"
    m_r_path = "./data/WN18RR/test2id_m_r.txt"

    m_f = _read_tsv(m_f_path)
    m_r = _read_tsv(m_r_path)

    m_f_r = m_f + m_r

    print(len(m_f))
    print(len(m_r))

    print(len(m_f_r))

    write_triples_2_id("./data/WN18RR/test2id_m_f_r.txt",m_f_r)


def read_obj(entity_obj_path):

    X = pd.read_csv(entity_obj_path,sep="\t",header=None)

    return np.array(X)


def test2id_split(dir_path,test_triple_path):
    
    test2id = _read_tsv(test_triple_path)
    n = 1
    sub_test_set = []
    for i in range(len(test2id)):
        
        sub_test_set.append(test2id[i])
        if len(sub_test_set) == 7000:
            write_triples_2_id(dir_path+"test_set_new2new2id_" + str(n) + ".txt", sub_test_set)
            sub_test_set = []
            n += 1
    
    write_triples_2_id(dir_path+"test_set_new2new2id_" + str(n) + ".txt", sub_test_set)
    
    
def gap_time(head_with_time, tail_with_time):

    format_pattern = "%Y-%m-%d %H:%M:%S"
    head_with_time = datetime.strptime(head_with_time, format_pattern)
    tail_with_time = datetime.strptime(tail_with_time, format_pattern)
    gap = (head_with_time - tail_with_time).total_seconds()
    gap_hour = round(gap / 3600, 3)
    print("gap_hour ",gap_hour)
    return  gap_hour


if __name__ == "__main__":


    # dict_set = {"a":1,"b":2,"c":3}
    # print(list(dict_set.keys()))
    # candidates = [1,2,3,4,5,6]
    # map_dandidates = map(lambda c: c[0], candidates)
    # print(map_dandidates)

    masterSetById = {}
    ss_st = masterSetById.get(1, set())
    # masterSetById.get(2, set())
    ss_st.add(2)
    print(masterSetById)



"""
candidates issue link about 2600- 3200 for MongoDB

1. half training triples don't have description

2. compare with Hadoop, Redhat, it has much more issue link candidates


600- 700 issue link candidates for Hadoop
500 - 600  issue link candidates for RedHat
"""


# time gap 14 days, 35% - 40% issue link are constructed within 14days
