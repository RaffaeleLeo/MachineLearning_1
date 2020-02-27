import decision_trees
import tree_helper
import pandas as pd
import numpy as np

all_classifiers = []
classifiers = []


def main():
    data = initialize_data(pd.read_csv('train.csv', header=None))
    print(np.power(4, 2))
    for x in range(0, 50):
        subset = data.sample(1000, replace=False)
        for p in range(0, 20):
            tree = decision_trees.start(subset, None, 17)
            classifiers.append(tree)
        all_classifiers.append(classifiers)
    calculate_final()


def calculate_final():
    training_data = initialize_data(pd.read_csv('train.csv', header=None))

    c = 0
    ic = 0
    big_V = 0
    big_B = 0
    for index, row in training_data.iterrows():
        count = 1

        variance = 0
        mean = 0
        for wl in all_classifiers:
            adda_guess = 0
            val = decision_trees.step_through_tree(wl[0], row)
            if val == 'yes':
                adda_guess = 1
                mean += 1
            else:
                adda_guess = -1
                mean -= 1
            if row.values[16] == 'yes':
                big_B += np.power(1 - adda_guess/50, 2)
            else:
                big_B += np.power(-1 - adda_guess/50, 2)
        for wl in all_classifiers:
            val = decision_trees.step_through_tree(wl[0], row)
            if val == 'yes':
                variance += 1 - mean/50
            else:
                variance += -1 - mean/50
        big_V += variance/49
    print('bias for single trees (total 50):')
    print('bias: ' + str(big_B/50))
    print('variance: ' + str(big_V/50))


def initialize_data(list_of_data):
    list_of_data.loc[list_of_data[0] <= tree_helper.att_threshold(list_of_data, 'age'), 0] = 0
    list_of_data.loc[list_of_data[0] > tree_helper.att_threshold(list_of_data, 'age'), 0] = 1

    list_of_data.loc[list_of_data[5] <= tree_helper.att_threshold(list_of_data, 'balance'), 5] = 0
    list_of_data.loc[list_of_data[5] > tree_helper.att_threshold(list_of_data, 'balance'), 5] = 1

    list_of_data.loc[list_of_data[9] <= tree_helper.att_threshold(list_of_data, 'day'), 9] = 0
    list_of_data.loc[list_of_data[9] > tree_helper.att_threshold(list_of_data, 'day'), 9] = 1

    list_of_data.loc[list_of_data[11] <= tree_helper.att_threshold(list_of_data, 'duration'), 11] = 0
    list_of_data.loc[list_of_data[11] > tree_helper.att_threshold(list_of_data, 'duration'), 11] = 1

    list_of_data.loc[list_of_data[12] <= tree_helper.att_threshold(list_of_data, 'campaign'), 12] = 0
    list_of_data.loc[list_of_data[12] > tree_helper.att_threshold(list_of_data, 'campaign'), 12] = 1

    list_of_data.loc[list_of_data[13] >= 0, 13] = 1
    list_of_data.loc[list_of_data[13] < 0, 13] = 0

    list_of_data.loc[list_of_data[14] <= tree_helper.att_threshold(list_of_data, 'previous'), 14] = 0
    list_of_data.loc[list_of_data[14] > tree_helper.att_threshold(list_of_data, 'previous'), 14] = 1

    return list_of_data


if __name__ == '__main__':
    main()

