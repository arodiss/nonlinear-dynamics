from __future__ import division
import abc
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


class AbstractMap():
    __metaclass__ = abc.ABCMeta

    def __init__(self, x0):
        self.t = [1]
        self.x = [x0]

    @abc.abstractmethod
    def generate_next_iterate(self):
        pass

    @abc.abstractmethod
    def generate_cobweb_base(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_min_k():
        pass

    @staticmethod
    @abc.abstractmethod
    def get_max_k():
        pass

    def get_last_x(self):
        return self.x[-1]

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

    def show_current_cobweb_plot(self, skip=50):
        plt.plot(self.x[skip:-1], self.x[skip+1:], color='red')
        xmin = min(self.x)
        xmax = max(self.x)
        xrange = [xmin + (xmax - xmin) * i / 100 for i in range(1, 100)]
        plt.plot(self.x[skip:-1], self.x[skip+1:], color='red')
        plt.plot(xrange, xrange, color='gray')
        plt.plot(*self.generate_cobweb_base(), color='blue')
        plt.legend(('Attractor', 'Generic constant x', 'Map-specific constant x'))
        plt.show()

    def generate_cobweb_plot(self, iterates=100, skip=50):
        for i in range(0, iterates):
            self.get_next_iterate()
        self.show_current_cobweb_plot(skip)

    def generate_time_plot(self, iterates=50):
        for i in range(0, iterates):
            self.get_next_iterate()
        self.show_current_time_plot()

    def get_bifurcation_portrait_slice(self, iterations_to_skip=1000, iterations_to_show=30):
        for i in range(0, iterations_to_skip + iterations_to_show):
            self.get_next_iterate()
        return self.x[-iterations_to_show:]

    @classmethod
    def generate_bifurcation_portrait(cls):
        xs = []
        ys = []
        for i in range(0, 10000):
            k = cls.get_min_k() + (cls.get_max_k() - cls.get_min_k()) * i / 10000
            map = cls(.2, k)
            for y in map.get_bifurcation_portrait_slice():
                xs.append(k)
                ys.append(y)
        plt.scatter(xs, ys, c='black', marker='.', s=.1)
        plt.xlabel('k')
        plt.ylabel('x distribution')
        plt.show()

    @classmethod
    def generate_reverse_bifurcation_portrait(cls):
        xs = []
        ys = []
        for i in range(0, 10000):
            k = cls.get_min_k() + (cls.get_max_k() - cls.get_min_k()) * i / 10000
            map = cls(.2, k)
            slice = map.get_bifurcation_portrait_slice(iterations_to_show=10)
            for x in slice:
                for y in slice:
                    xs.append(x)
                    ys.append(y)
        plt.scatter(xs, ys, c='black', marker='.', s=.1)
        plt.xlabel('x distribution')
        plt.ylabel('x distribution')
        plt.show()


    @classmethod
    def generate_reverse_bifurcation_frame(cls, k):
        xs = []
        ys = []
        for i in range(0, 100):
            map = cls(i / 100, k)
            slice = map.get_bifurcation_portrait_slice(iterations_to_show=10)
            for x in slice:
                for y in slice:
                    xs.append(x)
                    ys.append(y)
        fig = plt.figure()
        fig.add_subplot()
        plt.scatter(xs, ys, c='black', marker='.', s=.1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('k = {}'.format(k))
        plt.xlabel('x distribution')
        plt.ylabel('x distribution')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clear()
        plt.close()
        return frame


    @classmethod
    def generate_reverse_bifurcation_video(cls):
        out = cv2.VideoWriter(
            'out.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            20.0,
            (640, 640)
        )
        for i in tqdm(range(0, 2000)):
            k = cls.get_min_k() + (cls.get_max_k() - cls.get_min_k()) * i / 2000
            frame = cls.generate_reverse_bifurcation_frame(k)
            out.write(cv2.resize(frame, (640, 640)))
        out.release()


class LogisticMap(AbstractMap):
    def __init__(self, x0, r):
        super(LogisticMap, self).__init__(x0)
        self.r = r

    def generate_next_iterate(self):
        return self.r * self.get_last_x() * (1 - self.get_last_x())

    def generate_cobweb_base(self):
        xmin = min(self.x)
        xmax = max(self.x)
        xrange = [xmin + (xmax - xmin) * i / 100 for i in range(1, 100)]
        ys = []
        for x in xrange:
            ys.append(self.r * x - self.r * x * x)
        return xrange, ys

    @staticmethod
    def get_min_k():
        return 2

    @staticmethod
    def get_max_k():
        return 4


class QuadraticMap(AbstractMap):
    def __init__(self, x0, r):
        super(QuadraticMap, self).__init__(x0)
        self.r = r

    def generate_next_iterate(self):
        return (self.r * self.get_last_x() * self.get_last_x()) - 1

    def generate_cobweb_base(self):
        xmin = min(self.x)
        xmax = max(self.x)
        xrange = [xmin + (xmax - xmin) * i / 100 for i in range(1, 100)]
        ys = []
        for x in xrange:
            ys.append(self.r * x * x - 1)
        return xrange, ys

    @staticmethod
    def get_min_k():
        return 1

    @staticmethod
    def get_max_k():
        return 2


class TentMap(AbstractMap):
    def __init__(self, x0, r):
        super(TentMap, self).__init__(x0)
        self.r = r

    def generate_next_iterate(self):
        if self.get_last_x() < .5:
            return self.r * self.get_last_x()
        return self.r * (1 - self.get_last_x())

    def generate_cobweb_base(self):
        raise RuntimeError('Not implemented')

    @staticmethod
    def get_min_k():
        return 1

    @staticmethod
    def get_max_k():
        return 2


if __name__ == "__main__":
    LogisticMap.generate_reverse_bifurcation_portrait()


