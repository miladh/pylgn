import quantities as pq
import pylgn


def animate_cube(cube, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation        
    
    ims = []
    fig = plt.figure()
    for row in cube:
        ims.append([plt.imshow(row.real, animated=True)])
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    plt.colorbar()
    
    plt.title(title) if title is not None else 0

    plt.show()
