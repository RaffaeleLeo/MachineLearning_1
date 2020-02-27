# helper methods

import copy
import csv
import cmath
import pandas as pd
import decision_trees


def entropy_str(S, att, total):
    att_num = decision_trees.attributes_to_num[att]
    att_things = decision_trees.all_attributes[att]
    expected_entropy = 0
    for t in att_things:
        S_t = S.loc[S[att_num] == t]
        if len(S_t) > 0:
            len_yes = len(S_t.loc[S_t[16] == 'yes'])
            len_no = len(S_t.loc[S_t[16] == 'no'])
            if len_yes == 0:
                len_yes = 1
            if len_no == 0:
                len_no = 1

            expected_entropy += (len(S_t) / total) * \
                                (- (len(S_t.loc[S_t[16] == 'yes']) / len(S_t)) * (cmath.log(len_yes / len(S_t), 2))
                                 - (len(S_t.loc[S_t[16] == 'no']) / len(S_t)) * (cmath.log(len_no / len(S_t), 2))
                                 ).real
    return expected_entropy


def att_threshold(S, att):
    att_num = decision_trees.attributes_to_num[att]
    return S[att_num].median()


def is_att_low_or_high(S, att):
    att_num = decision_trees.attributes_to_num[att]
    threshold = S[att_num].median()

    if S[att_num] > threshold:
        return 'high'
    else:
        return 'low'
