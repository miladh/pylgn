import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
import quantities as pq

# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=7, nr=7, dt=2*pq.ms, dr=0.4*pq.deg)

# create kernels
Wg_r = spl.create_dog_ft()
Wg_t = tpl.create_biphasic_ft()

# create neurons
ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

# create stimulus
stimulus = pylgn.stimulus.create_natural_image(filenames="natural_scene.png",
                                               delay=40*pq.ms,
                                               duration=80*pq.ms)
network.set_stimulus(stimulus, compute_fft=True)

# compute
network.compute_response(ganglion)

import pylgn.tools as tls

# apply static nonlinearity and scale rates
rates = ganglion.response
rates = tls.heaviside_nonlinearity(rates)
rates = tls.scale_rates(rates, 60*pq.Hz)

# generate spike trains
spike_trains = tls.generate_spike_train(rates, integrator.times)

# visulize
pylgn.plot.animate_spike_activity(spike_trains,
                                  times=integrator.times,
                                  positions=integrator.positions,
                                  title="Spike activity")

pylgn.plot.raster_plot(spike_trains.flatten()[::200])  
