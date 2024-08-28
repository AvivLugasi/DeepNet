# This is a sample Python script.
import math

import cupy as cp


x = cp.random.rand(1, 100000)
print(cp.linalg.norm(x))
y = cp.array([0.0])
for i in range(0, x.shape[1]):
    y += x[0][i]*x[0][i]
print(cp.sqrt(y)[0])
print(cp.equal(cp.linalg.norm(x), cp.sqrt(y)[0]))

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
