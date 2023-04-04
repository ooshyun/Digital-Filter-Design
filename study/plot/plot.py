import numpy as np
import matplotlib.pyplot as plt

# One line show several graph
def show_one_line(x, resol=2**15):
    line = np.linspace(x[0], x[-1], resol, endpoint=True)
    y = np.ones_like(line)*np.nan
    for _x in x:
        buf = line - _x
        index = np.where(buf>=0)[0]
        y[index[0]] = line[index[0]]
    return line, y

# Annotation
# for x, y in zip(np.arange(len(mel_f)), mel_f):
#     plt.annotate(text=f"{str(x)}, {str(y)}", xy=(x+0.01*x, y+0.01*y))


x1 = np.linspace(0, 10, 10, endpoint=True)
x2 = np.linspace(0, 10, 100, endpoint=True)

line, x1 = show_one_line(x1)
line, x2 = show_one_line(x2)

# print(line.shape, x1.shape, x2.shape)

plt.scatter(line, x2)
plt.scatter(line, x1)
plt.show()