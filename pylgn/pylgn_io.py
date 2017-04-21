from abc import ABC, abstractmethod
from .core import closure_params

# TODO write doc


def _convert_defaultdict_list(list_dict):
    annotations = {}
    for key, value in list_dict.items():
        annotations[key] = {}
        for i, item in enumerate(value):
            annotations[key][str(key)+"_"+str(i)] = item

    print(annotations)
    return annotations


##############################################################
# BaseIO
##############################################################

class BaseIO(ABC):
    def __init__(self, filename):
        pass

    def write_network(self, network):
        self.write_integrator_params(network.integrator)
        self.write_stimulus_params(network.stimulus.closure)
        self.write_stimulus(network.stimulus)
        for neuron in network.neurons:
            self.write_neuron(neuron)

    @abstractmethod
    def write_neuron(self, neuron):
        pass

    @abstractmethod
    def write_stimulus(self, stimulus):
        pass

    @abstractmethod
    def write_stimulus_params(self, params):
        pass

    @abstractmethod
    def write_integrator_params(self, integrator):
        pass


##############################################################
# exdir IO
##############################################################

class ExdirIO(BaseIO):
    def __init__(self, filename):
        import exdir
        self.file = exdir.File(filename)

    def write_integrator_params(self, integrator):
        integrator_grp = self.file.require_group("integrator")
        integrator_grp.attrs["Nt"] = integrator.Nt
        integrator_grp.attrs["Nr"] = integrator.Nr
        integrator_grp.attrs["dt"] = integrator.dt
        integrator_grp.attrs["dr"] = integrator.dr
        integrator_grp.attrs["dw"] = integrator.dw
        integrator_grp.attrs["dk"] = integrator.dk

        integrator_grp.require_dataset("times", data=integrator.times)
        integrator_grp.require_dataset("positions", data=integrator.positions)
        integrator_grp.require_dataset("spatial_freqs", data=integrator.spatial_freqs)
        integrator_grp.require_dataset("temporal_freqs",
                                       data=integrator.temporal_freqs)

    def write_stimulus_params(self, params):
        stimulus_grp = self.file.require_group("stimulus")
        stimulus_grp.attrs = closure_params(params)

    def write_stimulus(self, stimulus):
        stimulus_grp = self.file.require_group("stimulus")
        if stimulus.spatiotemporal is not None:
            stimulus_grp.require_dataset("spatiotemporal", data=stimulus.spatiotemporal)

        if stimulus.ft is not None:
            stimulus_grp.require_dataset("fourier_transform", data=stimulus.ft)

    def write_neuron(self, neuron):
        from collections import defaultdict
        name = type(neuron).__name__.lower()
        neuron_grp = self.file.require_group(name)
        annotations = {}
        for key, value in neuron.annotations.items():
            if isinstance(value, defaultdict):
                annotations[key] = _convert_defaultdict_list(value)
            else:
                annotations[key] = value

        neuron_grp.attrs = annotations

        if neuron.response is not None:
            response_grp = neuron_grp.require_group("response")
            response_grp.require_dataset("spatiotemporal", data=neuron.response)

        if neuron.response_ft is not None:
            response_grp = neuron_grp.require_group("response")
            response_grp.require_dataset("fourier_transform", data=neuron.response_ft)

        if neuron.irf is not None:
            response_grp = neuron_grp.require_group("irf")
            response_grp.require_dataset("spatiotemporal", data=neuron.response)

        if neuron.irf_ft is not None:
            response_grp = neuron_grp.require_group("irf")
            response_grp.require_dataset("fourier_transform", data=neuron.response_ft)


##############################################################
# hdf5 IO
##############################################################
class Hdf5IO(BaseIO):
    # TODO: find a solution for dict attrs
    def __init__(self, filename):
        import h5py
        self.file = h5py.File(filename)

    def write_integrator_params(self, integrator):
        integrator_grp = self.file.require_group("integrator")
        integrator_grp.attrs["Nt"] = integrator.Nt
        integrator_grp.attrs["Nr"] = integrator.Nr
        integrator_grp.attrs["dt"] = integrator.dt
        integrator_grp.attrs["dr"] = integrator.dr
        integrator_grp.attrs["dw"] = integrator.dw
        integrator_grp.attrs["dk"] = integrator.dk

        integrator_grp.create_dataset("times", data=integrator.times)
        integrator_grp.create_dataset("positions", data=integrator.positions)
        integrator_grp.create_dataset("spatial_freqs", data=integrator.spatial_freqs)
        integrator_grp.create_dataset("temporal_freqs", data=integrator.temporal_freqs)

    def write_stimulus_params(self, params):
        stimulus_grp = self.file.require_group("stimulus")
        # for key, item in closure_params(params).items():
        #     stimulus_grp.attrs[key] = item

    def write_stimulus(self, stimulus):
        stimulus_grp = self.file.require_group("stimulus")
        if stimulus.spatiotemporal is not None:
            stimulus_grp.create_dataset("spatiotemporal", data=stimulus.spatiotemporal)

        if stimulus.ft is not None:
            stimulus_grp.create_dataset("fourier_transform", data=stimulus.ft)

    def write_neuron(self, neuron):
        name = type(neuron).__name__.lower()
        neuron_grp = self.file.require_group(name)
        # for key, item in neuron.params.items():
        #     print(key, item)
        #     neuron_grp.attrs[key] = item

        if neuron.response is not None:
            response_grp = neuron_grp.require_group("response")
            response_grp.create_dataset("spatiotemporal", data=neuron.response)

        if neuron.response_ft is not None:
            response_grp = neuron_grp.require_group("response")
            response_grp.create_dataset("fourier_transform", data=neuron.response_ft)

        if neuron.irf is not None:
            response_grp = neuron_grp.require_group("irf")
            response_grp.create_dataset("spatiotemporal", data=neuron.response)

        if neuron.irf_ft is not None:
            response_grp = neuron_grp.require_group("irf")
            response_grp.create_dataset("fourier_transform", data=neuron.response_ft)
