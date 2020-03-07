import pandas as pd
import numpy as np


def main():
    data = pd.read_csv('train.csv', header=None)

    r = 0.1

    weight_vector = [0, 0, 0, 0, 0]
    average_vector = [0, 0, 0, 0, 0]
    for T in range(0, 10):
        data = data.sample(frac=1).reset_index(drop=True)
        for index, row in data.iterrows():
            if (2*row[4] - 1) * (1*weight_vector[0] + row[0] * weight_vector[1] + row[1] * weight_vector[2] + row[2] * weight_vector[3] + row[3] * weight_vector[4]) <= 0:
                weight_vector[0] = weight_vector[0] + r*(2*row[4] - 1)*(1)
                weight_vector[1] = weight_vector[1] + r*(2*row[4] - 1)*(row[0])
                weight_vector[2] = weight_vector[2] + r * (2 * row[4] - 1) * (row[1])
                weight_vector[3] = weight_vector[3] + r * (2 * row[4] - 1) * (row[2])
                weight_vector[4] = weight_vector[4] + r * (2 * row[4] - 1) * (row[3])

            average_vector[0] += weight_vector[0]
            average_vector[1] += weight_vector[1]
            average_vector[2] += weight_vector[2]
            average_vector[3] += weight_vector[3]
            average_vector[4] += weight_vector[4]

    total = 0
    mistakes = 0
    print(average_vector)
    test_data = pd.read_csv('test.csv', header=None)
    for index, row in test_data.iterrows():
        prediction = np.sign(1*average_vector[0] + row[0] * average_vector[1] + row[1] * average_vector[2] + row[2] * average_vector[3] + row[3] * average_vector[4])
        total += 1
        if prediction != 2*row[4] - 1:
            mistakes += 1
    print(mistakes/total)


if __name__ == '__main__':
    main()