import quantities as pq
import pylgn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


class MidpointNormalize(colors.Normalize):
    """
    https://matplotlib.org/gallery/userdemo/colormap_normalizations.html
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def animate_cube(cube, title=None, dt=None,
                 vmin=None, vmax=None, cmap="RdBu_r",
                 save_anim=False, filename="anim.mp4", writer="ffmpeg"):
    """
    Animates 3d array

    Parameters
    ----------
    cube : quantity array/array_like
        input array (Nt x Nx x Ny)

    title : str, optional

    dt : quantity scalar, optional, default: None

    vmin : quantity scalar/float, optional, default: cube.min()

    vmin : quantity scalar/float, optional, default: cube.max()

    save_anim : bool, optional, default: False

    filename : str, optional, default: "anim.mp4"

    writer : str, optional, default: "ffmpeg"

    """
    fig = plt.figure()
    vmin = vmin or cube.min()
    vmax = vmax or cube.max()
    plt.title("") if title is None else plt.title(title)

    def init():
        im.set_data(cube[0, :, :])
        ttl.set_text("")
        return im, ttl

    def animate(j):
        im.set_data(cube[j, :, :])
        ttl.set_text("Frame = " + str(j)) if dt is None \
            else ttl.set_text("Time = {} {}".format(round(j*dt.magnitude, 2), 
                                                    dt.dimensionality))
        return im, ttl

    ttl = plt.suptitle("")
    im = plt.imshow(cube[0, :, :], animated=True, vmin=vmin, vmax=vmax,
                    origin="lower", cmap=cmap,
                    norm=MidpointNormalize(midpoint=0.))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=cube.shape[0], interval=50,
                                   repeat=True, repeat_delay=1000)

    plt.colorbar()
    if save_anim:
        anim.save(filename, writer=writer)
    plt.show()
