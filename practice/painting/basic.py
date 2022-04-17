import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

def f1():
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.
    plt.show()

def f2():
    np.random.seed(19680801)  # seed the random number generator.
    data = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.scatter('a', 'b', c='c', s='d', data=data)
    ax.set_xlabel('entry a')
    ax.set_ylabel('entry b')

    plt.show()

def f3():
    x = np.linspace(0, 2, 100)  # Sample data.

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
    ax.plot(x, x ** 3, label='cubic')  # ... and some more.
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()


def f4():
    data1, data2, data3, data4 = np.random.randn(4, 100)  # make 4 random data sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
    my_plotter(ax1, data1, data2, {'marker': 'x'})
    my_plotter(ax2, data3, data4, {'marker': 'o'})

    plt.show()

def f5():
    data1, data2, data3, data4 = np.random.randn(4, 100)
    fig, ax = plt.subplots(figsize=(5, 2.7))
    x = np.arange(len(data1))
    ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')
    l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)
    l.set_linestyle(':')

    plt.show()

def f6():
    data1, data2, data3, data4 = np.random.randn(4, 100)
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.scatter(data1, data2, s=50, facecolor='C0', edgecolor='k');

    plt.show()

def f7():
    data1, data2, data3, data4 = np.random.randn(4, 100)
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.plot(data1, 'o', label='data1')
    ax.plot(data2, 'd', label='data2')
    ax.plot(data3, 'v', label='data3')
    ax.plot(data4, 's', label='data4')
    ax.set_title(r'$\sigma_i=15$')
    ax.legend()

    plt.show()

def f8():
    fig, ax = plt.subplots(figsize=(5, 2.7))

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    line, = ax.plot(t, s, lw=2)

    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
                arrowprops=dict(facecolor='#FFCC99',edgecolor='None', shrink=0.01))

    ax.set_ylim(-2, 2)

    plt.show()

def f9():
    data1, data2, data3, data4 = np.random.randn(4, 100)
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.plot(np.arange(len(data1)), data1, label='jojo')
    ax.plot(np.arange(len(data2)), data2, label='dio')
    ax.plot(np.arange(len(data3)), data3, 'd', label='abdd')
    ax.legend()

    plt.show()

def f10():
    data1, data2, data3, data4 = np.random.randn(4, 100)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), layout='constrained')
    xdata = np.arange(len(data1))  # make an ordinal for this
    data = 10 ** data1
    axs[0].plot(xdata, data)

    axs[1].set_yscale('log')
    axs[1].plot(xdata, data)

    plt.show()

def f11():
    data1 = np.random.randn(100)
    xdata = np.arange(len(data1))

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(xdata, data1)
    axs[0].set_title('Automatic ticks')

    axs[1].plot(xdata, data1)
    axs[1].set_xticks(np.arange(0, 100, 30), ['zero', '30', 'sixty', '90'])
    axs[1].set_yticks([-1.5, 0, 1.5])  # note that we don't need to specify labels
    axs[1].set_title('Manual ticks')

    plt.show()

def f12():
    data1, data2, data3, data4 = np.random.randn(4, 100)
    X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
    Z = (1 - X / 2 + X ** 5 + Y ** 3) * np.exp(-X ** 2 - Y ** 2)

    fig, axs = plt.subplots(2, 2, layout='constrained')
    pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')
    fig.colorbar(pc, ax=axs[0, 0])
    axs[0, 0].set_title('pcolormesh()')

    co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
    fig.colorbar(co, ax=axs[0, 1])
    axs[0, 1].set_title('contourf()')

    pc = axs[1, 0].imshow(Z ** 2 * 100, cmap='plasma',
                          norm=mpl.colors.LogNorm(vmin=0.01, vmax=100))
    fig.colorbar(pc, ax=axs[1, 0], extend='both')
    axs[1, 0].set_title('imshow() with LogNorm()')

    pc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r')
    fig.colorbar(pc, ax=axs[1, 1], extend='both')
    axs[1, 1].set_title('scatter()')

    plt.show()

def f13():
    fig, axd = plt.subplot_mosaic([['upleft', 'right'],
                                   ['lowleft', 'right']], layout='constrained')
    axd['upleft'].set_title('upleft')
    axd['lowleft'].set_title('lowleft')
    axd['right'].set_title('right')

    plt.show()

def f14():

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()
    

if __name__ == '__main__':
    f14()