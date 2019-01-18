import matplotlib.pyplot as plt


class LogisticMap(object):
    def __init__(self, r, x0):
        self.r = r
        self.t = [1]
        self.x = [x0]

    def get_last_x(self):
        return self.x[-1]

    def generate_next_iterate(self):
        self.t.append(self.t[-1] + 1)
        self.x.append(self.r * self.get_last_x() * (1 - self.get_last_x()))
        return self.t[-1], self.get_last_x()

    def get_next_iterate(self):
        return self.generate_next_iterate()

    def show_current_time_plot(self):
        plt.scatter(self.t, self.x)
        plt.xlabel('# iterate')
        plt.ylabel('x')
        plt.ylim(0, 1)
        plt.show()

    def generate_time_plot(self, iterates=50):
        for i in range(0, iterates):
            self.generate_next_iterate()
        self.show_current_time_plot()


if __name__ == "__main__":
    logistic_map = LogisticMap(3.01, .2)
    logistic_map.generate_time_plot(100)
    print(logistic_map.get_last_x())
