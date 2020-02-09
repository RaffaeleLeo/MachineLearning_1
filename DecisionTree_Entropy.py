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
        best_attribute = train_using_entropy(S, attributes)
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
def train_using_entropy(S, attributes):
    # if attributes is None or len(attributes) == 0:
    #     return 0
    # print('--in entropy--')
    unacc = len(S.loc[S[6] == 'unacc'])
    acc = len(S.loc[S[6] == 'acc'])
    good = len(S.loc[S[6] == 'good'])
    vgood = len(S.loc[S[6] == 'vgood'])

    total = len(S)
    # print(total)
    if unacc == 0:
        unacc = 1
    if acc == 0:
        acc = 1
    if good == 0:
        good = 1
    if vgood == 0:
        vgood = 1

    total_entropy = (-(len(S.loc[S[6] == 'unacc'])/total)*(cmath.log(unacc/total, 4))
                     - (len(S.loc[S[6] == 'acc'])/total)*(cmath.log(acc/total, 4))
                     - (len(S.loc[S[6] == 'good'])/total)*(cmath.log(good/total, 4))
                     - (len(S.loc[S[6] == 'vgood'])/total)*(cmath.log(vgood/total, 4))
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
            if len(S_t) > 0:
                len_unacc = len(S_t.loc[S_t[6] == 'unacc'])
                len_acc = len(S_t.loc[S_t[6] == 'acc'])
                len_good = len(S_t.loc[S_t[6] == 'good'])
                len_vgood = len(S_t.loc[S_t[6] == 'vgood'])
                if len_unacc == 0:
                    len_unacc = 1
                if len_acc == 0:
                    len_acc = 1
                if len_good == 0:
                    len_good = 1
                if len_vgood == 0:
                    len_vgood = 1

                expected_entropy += (len(S_t)/total) * \
                                    (- (len(S_t.loc[S_t[6] == 'unacc'])/len(S_t))*(cmath.log(len_unacc/len(S_t), 4))
                                     - (len(S_t.loc[S_t[6] == 'acc'])/len(S_t))*(cmath.log(len_acc/len(S_t), 4))
                                     - (len(S_t.loc[S_t[6] == 'good'])/len(S_t))*(cmath.log(len_good/len(S_t), 4))
                                     - (len(S_t.loc[S_t[6] == 'vgood'])/len(S_t))*(cmath.log(len_vgood/len(S_t), 4))
                                     ).real

        IG = total_entropy - expected_entropy
        # print(IG)
        if IG > best_IG:
            best_attribute = a
            best_IG = IG
    # print('--out entropy--')
    return best_attribute


if __name__ == '__main__':
    main()
