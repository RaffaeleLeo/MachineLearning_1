from typing import List, Any

import decision_trees
import tree_helper
import pandas as pd
import numpy as np

global x_y_output
x_y_output = []

global weak_learners
weak_learners = []


def main():
    weights = pd.DataFrame(columns=['weights'])

    start_ensemble(initialize_data(pd.read_csv('train.csv', header=None)), weights, 5)


def start_ensemble(list_of_data, weights, T):
    weights = initialize_weights(list_of_data)
    list_of_data.insert(17, "weights", weights, True)
    while T > 0:
        # initialize weights
        # print(weights)
        # print(list_of_data)
        node = decision_trees.start(list_of_data, weights, 1)

        test_data = initialize_data(pd.read_csv('train.csv', header=None))

        # get the error of the tree
        error_of_tree = 0
        for index, row in test_data.iterrows():
            val = decision_trees.step_through_tree(node, row)
            if val != row.values[16]:
                error_of_tree += list_of_data['weights'][index]
                x_y_output[index] = -1
            else:
                x_y_output[index] = 1

        # get alpha_t
        alpha_t = .5 * np.log((1 - error_of_tree) / error_of_tree)
        #print(error_of_tree)
        # print('^^^^')
        z_t = 0
        for x in range(0, len(weights)):
            list_of_data.at[x, 'weights'] = list_of_data['weights'][x] * np.exp(-alpha_t * x_y_output[x])
            z_t += list_of_data['weights'][x]
        total = 0
        for x in range(0, len(weights)):
            list_of_data.at[x, 'weights'] = list_of_data['weights'][x] / z_t
            total += list_of_data['weights'][x]
        # print('^^^end^^^')
        #print(total)
        weak_learners.append([node, alpha_t, x_y_output])
        # if len(weak_learners) < 100 and len(weak_learners) % 10 == 0:
        #     calculate_final()
        # elif len(weak_learners) >= 100 and len(weak_learners) % 100 == 0:
        calculate_final()
        T -= 1


def initialize_weights(list_of_data):
    weights = []
    x_y_output.clear()
    weak_learners.clear()
    for x in range(0, len(list_of_data)):
        weights.append(1/len(list_of_data))
        x_y_output.append(1)
    return weights


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


def calculate_final():
    training_data = initialize_data(pd.read_csv('test.csv', header=None))
    testing_data = initialize_data(pd.read_csv('test.csv', header=None))

    c = 0
    ic = 0
    for index, row in training_data.iterrows():
        count = 1
        adda_guess = 0
        for wl in weak_learners:
            val = decision_trees.step_through_tree(wl[0], row)
            if val == 'yes':
                # print('ok?')
                adda_guess += wl[1]
            else:
                adda_guess -= wl[1]
            # print('guess after ' + str(count) + ' trees')
            # print(adda_guess)
            count += 1
        # if row.values[16] == 'yes':
        #     print(np.sign(adda_guess))
        if (np.sign(adda_guess) == 1.0 and row.values[16] == 'yes') or \
           (np.sign(adda_guess) == -1.0 and row.values[16] == 'no'):
            c += 1
        else:
            ic += 1

    print('percent correct ' + str(len(weak_learners)) + ':')
    print(c/(c+ic))
    print('fin')


if __name__ == '__main__':
    main()