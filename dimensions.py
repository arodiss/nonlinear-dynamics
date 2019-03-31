from __future__ import division
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log
from sklearn.linear_model import TheilSenRegressor


# too slow, not used
class Box(object):
    def __init__(self, floors, epsilons):
        ranges = []
        for index, floor in enumerate(floors):
            ranges.append((floor, floor + epsilons[index]))
        self.ranges = ranges

    def contains(self, point):
        matches = True
        for index, coordinate in enumerate(point):
            matches = matches and self.ranges[index][1] >= coordinate >= self.ranges[index][0]
        return matches


def make_boxes(dimension_maxes, epsilons):
    floors = []
    for dim_index, dimension_max in enumerate(dimension_maxes):
        floors.append(list(np.arange(0, dimension_max, epsilons[dim_index])))

    floor_combos = [[f] for f in floors[0]]
    for i in range(1, len(floors)):
        new_floor_combos = []
        for old_floor_combo in floor_combos:
            for new_value in floors[i]:
                new_floor_combos.append(old_floor_combo + [new_value])
        floor_combos = new_floor_combos

    return floor_combos


def box_contains(floors, epsilons, point):
    for index, coordinate in enumerate(point):
        if not (floors[index] + epsilons[index] >= coordinate >= floors[index]):
            return False
    return True


def count_boxes(points, all_floors, epsilons):
    boxes_filled = [0] * len(all_floors)
    for point in tqdm(points):
        found = False
        for box_index, floors in enumerate(all_floors):
            if box_contains(floors, epsilons, point):
                boxes_filled[box_index] = 1
                found = True
                break
        if not found:
            print('Warning: cannot match point {} to any box'.format(point))

    return sum(boxes_filled)


def get_capacity_dimension(data):
    # assumes 3 dims
    minx = data[0].min()
    maxx = data[0].max()
    miny = data[1].min()
    maxy = data[1].max()
    minz = data[2].min()
    maxz = data[2].max()
    
    points = [[row[0] - minx, row[1] - miny, row[2] - minz] for i, row in data.iterrows()]  # assumes 3 dims

    results = []
    possible_epsilons = list(np.arange(35, .2, -.05))
    for i, epsilon in enumerate(possible_epsilons):
        print('Counting with epsilon {} (iteration {}/{})'.format(epsilon, i+1, len(possible_epsilons)))
        epsilons = [epsilon, epsilon, epsilon]  # assumes 3 dims
        all_floors = make_boxes([maxx - minx, maxy - miny, maxz - minz], epsilons)  # assumes 3 dims
        results.append({
            'epsilon': epsilon,
            'num_boxes': len(all_floors),
            'filled_boxes': count_boxes(points, all_floors, epsilons)
        })
    return results


if __name__ == "__main__":
    data = pd.read_csv('CapDimData.dat', header=None)
    data = get_capacity_dimension(data)
    print(data)
    y = [log(i['filled_boxes']) for i in data]
    x = [log(1 / i['epsilon']) for i in data]
    regressor = TheilSenRegressor(random_state=42)
    regressor.fit(np.array(x)[:, np.newaxis], y)
    print(regressor.coef_)

    plt.plot(x, y)
    plt.plot(x, [regressor.predict(xx) for xx in x], color='red')
    plt.xlabel('log(1/epsilon)')
    plt.ylabel('log(num boxes)')
    plt.legend(['Data', 'Fit: slope {:.2}'.format(regressor.coef_[0])])
    plt.show()


    # 2 dims - slope 1.7
