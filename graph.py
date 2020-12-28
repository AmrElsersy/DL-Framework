import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
import time

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()

def live():

    def animate(i):
        plt.cla()
        plt.plot(range(len(y_vals)), y_vals)

    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()


Thread(target=live, daemon=True).start()

while 1:
    y_vals.append(random.randint(0, 5))
    time.sleep(1)
    print(x_vals)
    print(y_vals)
