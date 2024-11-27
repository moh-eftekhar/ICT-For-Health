import numpy as np


def most_frequent(x):
    return np.bincount(x).argmax()


def mapping(input):
    actions = 19
    km_mapping = []
    spilited = np.array_split(input, actions)
    for i in range(actions):
        km_mapping.append(most_frequent(spilited[i]))
    return km_mapping


def remap(input):
    j_temp = []
    for i in range(19):
        if i not in j_temp:
            element = input[i]
            input[i] = i + 1
        for j in range(i + 1, 19):
            if input[j] == element:
                input[j] = i + 1
                j_temp.append(j)

    return input
