import quantities as pq
import pylgn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def animate_cube(cube, title=None, dt=None, save_anim=False):
    fig = plt.figure()
    vmin = cube.min()
    vmax = cube.max()
    plt.title("") if title is None else plt.title(title)

    def init():
        im.set_data(cube[0, :, :])
        ttl.set_text("")
        return im, ttl

    def animate(j):
        im.set_data(cube[j, :, :])
        ttl.set_text("Frame = " + str(j)) if dt is None \
            else ttl.set_text("Time = " + str('%.1f' % (j*dt.magnitude,)) + str(dt.units).split(" ")[-1])
        return im, ttl

    ttl = plt.suptitle("")
    im = plt.imshow(cube[0, :, :], animated=True, vmin=vmin, vmax=vmax,
                    origin="lower", cmap="RdBu_r",
                    norm=MidpointNormalize(midpoint=0.))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=cube.shape[0], interval=50,
                                   repeat=True, repeat_delay=1000)

    plt.colorbar()
    if save_anim:
        anim.save('im.mp4', writer="ffmpeg")
    plt.show()
