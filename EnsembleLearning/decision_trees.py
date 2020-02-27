# Raffaele Leo, U1089942
import copy
import csv
import cmath
import pandas as pd
import tree_helper
import numpy as np


class NodeBody:
    def __init__(self):
        self.label = ''
        self.nodes = {}

    def add_label(self, label):
        self.label = label

    def add_node(self, rout, next_node):
        self.nodes.update( {rout : next_node} )


class LeafNode:
    def __init__(self, attribute):
        self.attribute = attribute


global all_attributes
all_attributes = {'age': [0, 1],
                  'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                          'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
                  'marital': ['married', 'divorced', 'single'],
                  'education': ['unknown', 'secondary', 'primary', 'tertiary'],
                  'default': ['yes', 'no'],
                  'balance': [0, 1],
                  'housing': ['yes', 'no'],
                  'loan': ['yes', 'no'],
                  'contact': ['unknown', 'telephone', 'cellular'],
                  'day': [0, 1],
                  'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                  'duration': [0, 1],
                  'campaign': [0, 1],
                  'pdays': [0, 1],
                  'previous': [0, 1],
                  'poutcome': ['unknown', 'other', 'failure', 'success']}

global attributes_to_num
attributes_to_num = {'age': 0, 'job': 1, 'marital': 2, 'education': 3, 'default': 4, 'balance': 5, 'housing': 6,
                     'loan': 7, 'contact': 8, 'day': 9, 'month': 10, 'duration': 11, 'campaign': 12, 'pdays': 13,
                     'previous': 14, 'poutcome': 15}


def start(list_of_data, weights, tree_size):

    node = ID3(list_of_data,
               ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'],
               ['yes', 'no'],
               tree_size,
               weights)

    return node


def step_through_tree(tree, row):
    if len(tree.nodes) == 0:
        return tree.label
    att_num = attributes_to_num[tree.label]
    if row.values[att_num] in tree.nodes:
        return step_through_tree(tree.nodes[row.values[att_num]], row)
    else:
        return tree.label


def ID3(S, attributes, label, depth, weights):
    if attributes is None or len(attributes) == 0 or depth == 0:
        if weights is None:
            A = S[16].mode()
            node = NodeBody()
            if len(A) == 1:
                node.label = A.item()
            else:
                node.label = A[0]
            return node
        else:
            no_weight = S.loc[S[16] == 'no']['weights'].sum()
            yes_weight = S.loc[S[16] == 'yes']['weights'].sum()

            # print(A.item())
            node = NodeBody()
            if yes_weight > no_weight:
                node.label = 'yes'
            else:
                node.label = 'no'
            return node
    if S[16].nunique() == 1:
        # print('UNIQUE')
        # print(S[6])
        node = NodeBody()
        node.label = S[16].values[0]
        return node
    else:
        # start the head nodes
        head_node = NodeBody()
        # get the best attribute to split on
        if weights is None:
            best_attribute = train_using_entropy(S, attributes)
        else:
            best_attribute = train_using_weights(S, attributes, weights)
        attributes.remove(best_attribute)
        # get the number to locate that attribute in S
        att_num = attributes_to_num[best_attribute]
        # get a list of the types that that attribute can take
        attribute_types = all_attributes[best_attribute]
        # attain the subset for each attribute type
        depth -= 1
        for v in attribute_types:
            # start the subset
            s_v = S.loc[S[att_num] == v]
            if len(s_v) == 0:
                # CHANGE THIS CODE YOU MONSTER
                A = S[16].mode()
                node = NodeBody()
                if len(A) == 1:
                    node.label = A.item()
                else:
                    node.label = A[0]
                return node
            else:
                head_node.label = best_attribute
                a = copy.deepcopy(attributes)
                head_node.add_node(v, ID3(s_v, a, label, depth, weights))
        return head_node


def train_using_weights(S, attributes, weights):
    # if attributes is None or len(attributes) == 0:
    #     return 0
    # print('--in entropy--')

    no_total = S.loc[S[16] == 'no']['weights'].sum()
    yes_total = S.loc[S[16] == 'yes']['weights'].sum()

    total = no_total + yes_total
    # print('-----')
    # print(S)
    # print(yes_total)
    # print(no_total)
    total_entropy = (-yes_total/total * (cmath.log(yes_total/total, 2))
                     - no_total/total * (cmath.log(no_total/total, 2))
                     ).real
    # print(total_entropy)
    best_attribute = attributes[0]
    best_IG = 0
    for a in attributes:
        att_num = attributes_to_num[a]
        att_things = all_attributes[a]
        expected_entropy = 0
        for t in att_things:
            S_t = S.loc[S[att_num] == t]

            no_total_t = S_t.loc[S_t[16] == 'no']['weights'].sum()

            yes_total_t = S_t.loc[S_t[16] == 'yes']['weights'].sum()
            s_t_total = yes_total_t + no_total_t
            if no_total_t == 0:
                no_total_t = 0.0000001
            if yes_total_t == 0:
                yes_total_t = 0.00000001
            expected_entropy += (s_t_total / total) * \
                                (- no_total_t/s_t_total * (cmath.log(no_total_t/s_t_total, 2))
                                 - yes_total_t/s_t_total * (cmath.log(yes_total_t/s_t_total, 2))
                                 ).real

        IG = total_entropy - expected_entropy
        # print(IG)
        if IG > best_IG:
            best_attribute = a
            best_IG = IG
    # print('--out entropy--')
    return best_attribute


# Function to perform training with entropy.
def train_using_entropy(S, attributes):
    # if attributes is None or len(attributes) == 0:
    #     return 0
    # print('--in entropy--')
    yes = len(S.loc[S[16] == 'yes'])
    no = len(S.loc[S[16] == 'no'])

    total = len(S)
    # print(total)
    if yes == 0:
        yes = 1
    if no == 0:
        no = 1

    total_entropy = (-(len(S.loc[S[16] == 'yes'])/total)*(cmath.log(yes/total, 2))
                     - (len(S.loc[S[16] == 'no'])/total)*(cmath.log(no/total, 2))
                     ).real
    # print(total_entropy)
    best_attribute = attributes[0]
    best_IG = 0
    for a in attributes:
        expected_entropy = tree_helper.entropy_str(S, a, total)

        IG = total_entropy - expected_entropy
        # print(IG)
        if IG > best_IG:
            best_attribute = a
            best_IG = IG
    # print('--out entropy--')
    return best_attribute
