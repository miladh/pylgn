import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
import quantities as pq

# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=8, nr=9, dt=1*pq.ms, dr=0.1*pq.deg)

# create kernels
Wg_r = spl.create_dog_ft()
Wg_t = tpl.create_biphasic_ft()

# create neurons
ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

# create stimulus
stimulus = pylgn.stimulus.create_natural_image(filename="natural_scene.png",
                                               delay=40*pq.ms,
                                               duration=80*pq.ms)
network.set_stimulus(stimulus, compute_fft=True)

# compute
network.compute_response(ganglion)

# visulize
pylgn.plot.animate_cube(ganglion.response,
                        title="Ganglion cell responses",
                        dt=integrator.dt.rescale("ms"))
