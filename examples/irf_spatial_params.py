import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=5, nr=7, dt=1, dr=1)

# create neurons
ganglion = network.create_ganglion_cell()
relay = network.create_relay_cell()
cortical = network.create_cortical_cell()

# create kernels
Krg_r = spl.create_gauss_ft()
Krg_t = tpl.create_delta_ft()

# connect neurons
network.connect(ganglion, relay, (Krg_r, Krg_t))
network.connect(relay, cortical, (Krg_r, Krg_t))

# create stimulus
k_g = integrator.spatial_freqs[3]
w_g = integrator.temporal_freqs[1]
stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=w_g,
                                                      wavenumber=k_g,
                                                      orient=90.0)
network.set_stimulus(stimulus)
print(pylgn.closure_params(stimulus))
# compute
network.compute_response(relay)
network.compute_response(cortical)

print("\n", relay.annotations)
# visulize
print(relay.response.shape)
pylgn.plot.animate_cube(cortical.response, title="Relay cell response")
