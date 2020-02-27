import edited_trees
import tree_helper
import pandas as pd
import numpy as np

all_classifiers = []
classifiers = []


def main():
    data = initialize_data(pd.read_csv('train.csv', header=None))

    for x in range(0, 200):
        subset = data.sample(len(data), replace=True)
        # for p in range(0, 50):
        tree = edited_trees.start(subset, None, 17)
        classifiers.append(tree)
        # all_classifiers.append(classifiers)

        if len(classifiers) % 10 == 0:
            calculate_final()


def calculate_final():
    training_data = initialize_data(pd.read_csv('test.csv', header=None))

    c = 0
    ic = 0
    for index, row in training_data.iterrows():
        count = 1
        adda_guess = 0
        for wl in classifiers:
            val = edited_trees.step_through_tree(wl, row)
            if val == 'yes':
                adda_guess += 1
            else:
                adda_guess -= 1
            count += 1
        if (np.sign(adda_guess) > 0 and row.values[16] == 'yes') or \
           (np.sign(adda_guess) < 0 and row.values[16] == 'no'):
            c += 1
        else:
            ic += 1

    print('percent correct ' + str(len(classifiers)) + ':')
    print(c/(c+ic))


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

