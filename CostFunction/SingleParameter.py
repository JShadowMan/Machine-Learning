#!/usr/bin/env python3
#
# Copyright (C) 2017 Jayson_Wang
#
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


ROOM_COUNT = 5


def print_boston_data(line_count: int = 8) -> None:
    _boston_data = datasets.load_boston()

    for feature_name in _boston_data.feature_names:
        print('{:^8}'.format(feature_name), end='\t')
    else:
        print("\n", end='')

    for line in _boston_data.data[:line_count]:
        for column in line:
            print('{:^8}'.format(column), end='\t')
        else:
            print("\n", end='')


def draw_room_data():
    boston_data = datasets.load_boston()
    data, target = boston_data.data, boston_data.target

    room_data = [d[ROOM_COUNT] for d in data]

    figure = plt.figure()

    h_plot = figure.add_subplot(3, 1, 1)
    h_plot.scatter(x=room_data, y=target, s=1)

    # j(x) = θ1X, single parameter
    x_range = np.arange(0, 7, .01)
    distances = list()
    for x in x_range:
        square_sum = 0
        for i, room_count in enumerate(room_data):
            square_sum += np.power(x * room_count - target[i], 2)
        distances.append(square_sum / 2 * len(room_data))

    j_plot = figure.add_subplot(3, 1, 2)
    j_plot.scatter(x=x_range, y=distances, s=1, c='#ff0000')

    position = distances.index(np.min(distances))
    x_value = x_range[position]

    line_space = np.linspace(np.min(room_data), np.max(room_data), 1000)
    h_range = [room_count * x_value for room_count in line_space]
    d_plot = figure.add_subplot(3, 1, 3)
    d_plot.scatter(x=room_data, y=target, s=1, c='#00ff00')
    d_plot.scatter(x=line_space, y=h_range, s=1, c='#0000ff', marker='_')

    plt.show()


if __name__ == '__main__':
    # print_boston_data()
    draw_room_data()
