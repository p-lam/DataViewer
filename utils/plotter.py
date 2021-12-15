import numpy as np


# For plotting sets of data that share an x axis on separate axes (plots them round robin by index)
class CommonX:
    def __init__(self, axes, xAxis, xLabel=None):
        self.i = 0
        self.axes = axes
        self.xAxis = xAxis
        self.xLabel = xLabel
        self.it = 0
        pass

    def plot(self, values, label=None, title=None):
        ax = self.axes[self.i]
        ax.plot(self.xAxis, values)
        if self.xLabel is not None: ax.set_xlabel(self.xLabel)
        if label is not None: ax.set_ylabel(label)
        if title is not None: ax.set_title(title)
        self.i = (self.i + 1) % len(self.axes)


# For plotting scatter-plots with trend-lines and correlations
class Scatter_And_Trend:
    def __init__(self, xLabel=None, yLabel=None, title=None):
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.title = title

    def plot(self, axis, x, y):
        if self.xLabel is not None: axis.set_xlabel(self.xLabel)
        if self.yLabel is not None: axis.set_ylabel(self.yLabel)
        if self.title is not None: axis.set_title(self.title)
        axis.scatter(x, y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axis.plot(x, p(x))
        correlation = np.corrcoef(x, y)[0, 1]
        axis.text(x[0], p(x)[0] * 1.1, "R={:.3f}, Rsq={:.3f}".format(correlation, correlation ** 2))
