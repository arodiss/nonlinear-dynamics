from __future__ import division
import abc
import matplotlib.pyplot as plt


class AbstractMap():
    __metaclass__ = abc.ABCMeta

    def __init__(self, x0):
        self.t = [1]
        self.x = [x0]

    def get_last_x(self):
        return self.x[-1]

    @abc.abstractmethod
    def generate_next_iterate(self):
        pass

    def get_next_iterate(self):
        self.t.append(self.t[-1] + 1)
        self.x.append(self.generate_next_iterate())
        return self.t[-1], self.get_last_x()

    def show_current_time_plot(self):
        plt.scatter(self.t, self.x)
        plt.xlabel('# iterate')
        plt.ylabel('x')
        plt.ylim(0, 1)
        plt.show()

    def generate_time_plot(self, iterates=50):
        for i in range(0, iterates):
            self.get_next_iterate()
        self.show_current_time_plot()

    def get_bifurcation_portrait_slice(self, iterations_to_skip=1000, iterations_to_show=100):
        for i in range(0, iterations_to_skip + iterations_to_show):
            self.get_next_iterate()
        return self.x[-iterations_to_show:]


class LogisticMap(AbstractMap):
    def __init__(self, x0, r):
        super(LogisticMap, self).__init__(x0)
        self.r = r

    def generate_next_iterate(self):
        return self.r * self.get_last_x() * (1 - self.get_last_x())


class QuadraticMap(AbstractMap):
    def __init__(self, x0, r):
        super(QuadraticMap, self).__init__(x0)
        self.r = r

    def generate_next_iterate(self):
        return (self.r * self.get_last_x() * self.get_last_x()) - 1


if __name__ == "__main__":
    # logistic_map = LogisticMap(.3, 3.6)
    # logistic_map.generate_time_plot(1000)
    # print(logistic_map.get_last_x())

    xs = []
    ys = []
    for i in range(0, 1000):
        k = 2.9 + .9 * i / 1000
        quadratic_map = LogisticMap(.2, k)
        for y in quadratic_map.get_bifurcation_portrait_slice():
            xs.append(k)
            ys.append(y)
    plt.scatter(xs, ys, c='black', marker='.')
    plt.xlabel('k')
    plt.ylabel('x distribution')
    plt.show()
