# Raffaele Leo, U1089942

import csv
import cmath
import pandas as pd


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
all_attributes = {'buying':['low', 'med', 'high', 'vhigh'], 'maint':['low', 'med', 'high', 'vhigh'],
                  'doors':['2', '3', '4', '5more'], 'persons':['2', '4', 'more'], 'lug_boot':['small', 'med', 'big'],
                  'state': ['low', 'med', 'high']}

global attributes_to_num
attributes_to_num = {'buying': 0, 'maint': 1, 'doors': 2, 'persons': 3, 'lug_boot': 4, 'state': 5}


def main():
    list_of_data = pd.read_csv('train.csv', header=None)
    node = ID3(list_of_data, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'state'], ['unacc', 'acc','good', 'vgood'])

    test_data = pd.read_csv('test.csv', header=None)
    correct = 0
    incorrect = 0
    for index, row in test_data.iterrows():
        val = step_through_tree(node, row)
        if val == row.values[6]:
            correct += 1
        else:
            incorrect += 1
    print('correct: ' + str(correct))
    print('incorrect: ' + str(incorrect))
    pc = correct / (correct + incorrect)
    print('percent correct: ' + str(pc))


def step_through_tree(tree, row):
    if len(tree.nodes) == 0:
        return tree.label
    att_num = attributes_to_num[tree.label]
    if row.values[att_num] in tree.nodes:
        return step_through_tree(tree.nodes[row.values[att_num]], row)
    else:
        return tree.label


def ID3(S, attributes, label):
    if attributes is None or len(attributes) == 0:
        A = S[6].mode()
        # print(A.item())
        node = NodeBody()
        node.label = A.item()
        return node
    if S[6].nunique() == 1:
        # print('UNIQUE')
        # print(S[6])
        node = NodeBody()
        node.label = S[6].values[0]
        return node
    else:
        # start the head nodes
        head_node = NodeBody()
        # get the best attribute to split on
        best_attribute = train_using_ME(S, attributes)
        attributes.remove(best_attribute)
        # get the number to locate that attribute in S
        att_num = attributes_to_num[best_attribute]
        # get a list of the types that that attribute can take
        attribute_types = all_attributes[best_attribute]
        # attain the subset for each attribute type
        for v in attribute_types:
            # start the subset
            s_v = S.loc[S[att_num] == v]
            if len(s_v) == 0:
                # CHANGE THIS CODE YOU MONSTER
                A = S[6].mode()
                #print(A.item())
                node = NodeBody()
                node.label = A.item()
                return node
            else:
                head_node.label = best_attribute
                head_node.add_node(v, ID3(s_v, attributes, label))
        return head_node


# Function to perform training with entropy.
def train_using_ME(S, attributes):
    # if attributes is None or len(attributes) == 0:
    #     return 0
    # print('--in entropy--')
    items_counts = S[6].value_counts()
    max_item = items_counts.max()

    total = len(S)

    total_ME = (total - max_item)/total
    # print(total_ME)
    best_attribute = attributes[0]
    best_IG = 0
    for a in attributes:
        att_num = attributes_to_num[a]
        att_things = all_attributes[a]
        expected_ME = 0
        for t in att_things:
            S_t = S.loc[S[att_num] == t]
            if len(S_t) > 0:
                items_counts = S_t[6].value_counts()
                max_item_t = items_counts.max()
                expected_ME += (len(S_t)/total) * ((len(S_t) - max_item_t) / len(S_t))

        IG = total_ME - expected_ME
        # print(IG)
        if IG > best_IG:
            best_attribute = a
            best_IG = IG
    # print('--out entropy--')
    return best_attribute


if __name__ == '__main__':
    main()
