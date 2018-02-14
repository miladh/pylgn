import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import quantities as pq


def _unpack_kernel_tuple(kernel):
    from inspect import signature

    if not isinstance(kernel, tuple):
        raise TypeError("kernel is not a tuple", type(kernel))

    if "w" in str(signature(kernel[0])):
        temporal, spatial = kernel[0], kernel[1]
        spatiotemporal_kernel = lambda w, kx, ky: temporal(w) * spatial(kx, ky)
    else:
        temporal, spatial = kernel[1], kernel[0]
        spatiotemporal_kernel = lambda w, kx, ky: temporal(w) * spatial(kx, ky)

    return spatiotemporal_kernel


def closure_params(closure):
    """
    Stores closure parameters in a dict

    Parameters
    ----------
    closure : function
            A closure function

    Returns
    -------
    out : dict
        Dictionary

    """
    attrs = {}

    for cell_name, cell in zip(closure.__code__.co_freevars,
                               closure.__closure__):
        cell_contents = cell.cell_contents
        if not callable(cell_contents):
            if "params" not in attrs:
                attrs["params"] = {}
            attrs["params"][cell_name] = cell_contents
            attrs["type"] = closure.__qualname__.split(".")[0]
        else:
            attrs[cell_name] = closure_params(cell_contents)
            attrs[cell_name]["type"] = cell_contents.__qualname__.split(".")[0]
    return attrs


##############################################################
# Integrator class
##############################################################
class Integrator:
    """
    Integrator class for fast Fourier transform calculations.

    Attributes
    ----------
    times
    positions
    temporal_angular_freqs
    spatial_angular_freqs
    Nt : int
         Number of spatial points.
    Nr : int
         Number of temporal points
    dt : quantity scalar
         Temporal resolution
    dr : quantity scalar
         Spatial resolution
    dw : quantity scalar
         Temporal frequency resolution
    dk : quantity scalar
         Spatial frequency resolution
    """
    def __init__(self, nt, nr, dt, dr):
        """
        Integrator constructor

        Parameters
        ----------
        nt : int
             The power to raise 2 to. Number of temporal points is 2**nt.
        nr : int
             The power to raise 2 to. Number of spatial points is 2**nr.
        dt : quantity scalar
             Temporal resolution
        dr : quantity scalar
             Spatial resolution
        """

        self.Nt = int(2**nt)
        self.Nr = int(2**nr)
        self.dt = dt.rescale("ms") if isinstance(dt, pq.Quantity) else dt * pq.ms
        self.dr = dr.rescale("deg") if isinstance(dr, pq.Quantity) else dr * pq.deg
        self.w_s = 2 * np.pi / self.dt
        self.k_s = 2 * np.pi / self.dr
        self.dw = self.w_s / self.Nt
        self.dk = self.k_s / self.Nr

        self._t_vec = np.linspace(0, self.Nt - 1, self.Nt) * self.dt
        self._r_vec = np.linspace(-self.Nr/2, self.Nr/2 - 1, self.Nr) * self.dr
        self._w_vec = np.fft.fftfreq(self.Nt, self.dt) * -2 * np.pi
        self._k_vec = np.fft.fftfreq(self.Nr, self.dr) * 2 * np.pi

        self._fft_factor = self.dt.magnitude * self.dr.magnitude**2
        self._ifft_factor = 1. / self._fft_factor

    def meshgrid(self):
        """
        Spatiotemporal meshgrid

        Returns
        -------
        out : t_vec, y_vec, x_vec: quantity arrays
            time, x, and y values.

        """
        return np.meshgrid(self._t_vec, self._r_vec, self._r_vec,
                           indexing='ij', sparse=True)

    def freq_meshgrid(self):
        """
        Frequency meshgrid

        Returns
        -------
        out : w_vec, ky_vec, kx_vec: quantity arrays
            temporal and spatial frequency values.

        """
        return np.meshgrid(self._w_vec, self._k_vec,
                           np.abs(self._k_vec[:int(self.Nr/2+1)]),
                           indexing='ij', sparse=True)

    @property
    def times(self):
        """
        Time array

        Returns
        -------
        out : quantity array
            times

        """
        return self._t_vec

    @property
    def positions(self):
        """
        Position array

        Returns
        -------
        out : quantity array
            positions

        """
        return self._r_vec

    @property
    def temporal_angular_freqs(self):
        """
        Temporal angular frequency array

        Returns
        -------
        out : quantity array
            temporal angular frequencies

        """
        return self._w_vec.rescale(pq.kHz)

    @property
    def temporal_freqs(self):
        """
        Temporal frequency array

        Returns
        -------
        out : quantity array
            temporal frequencies

        """
        return self.temporal_angular_freqs / 2 / np.pi

    @property
    def spatial_angular_freqs(self):
        """
        Spatial angular frequency array

        Returns
        -------
        out : quantity array
            spatial angular frequencies

        """
        return self._k_vec

    @property
    def spatial_freqs(self):
        """
        Spatial frequency array

        Returns
        -------
        out : quantity array
            spatial frequencies

        """
        return self.spatial_angular_freqs / 2 / np.pi

    def compute_inverse_fft(self, cube):
        """
        Computes inverse fast Fourier transform.

        Parameters
        ----------
        cube : array_like
             input array (3-dimensional)


        Returns
        -------
        out : array_like
            transformed array

        """
        cube = np.fft.irfftn(cube)
        return np.fft.fftshift(cube, axes=(1, 2)) * self._ifft_factor

    def compute_fft(self, cube):
        """
        Computes fast Fourier transform.

        Parameters
        ----------
        cube : array_like
             input array (3-dimensional)


        Returns
        -------
        out : array_like
            transformed array

        """
        cube = np.fft.rfftn(np.fft.fftshift(cube, axes=(1, 2)))
        return cube * self._fft_factor


##############################################################
# Network
##############################################################
class Network:
    """
    Network class

    Attributes
    ----------
    neurons : list
         List with pylgn.Neuron objects
    integrator : pylgn.Integrator
         Integrator object
    stimulus : pylgn.Stimulus
         Stimulus object
    """
    def __init__(self, memory_efficient=False):
        """
        Network constructor
        """
        self.neurons = []
        self.stimulus = None
        self.memory_efficient = memory_efficient

    def create_integrator(self, nt, nr, dt, dr):
        """
        Create and set integrator

        Parameters
        ----------
        nt : int
             The power to raise 2 to. Number of temporal points is 2**nt.
        nr : int
             The power to raise 2 to. Number of spatial points is 2**nr.
        dt : quantity scalar
             Temporal resolution
        dr : quantity scalar
             Spatial resolution

        Returns
        -------
        out : pylgn.Integrator
            Integrator object
        """
        self.integrator = Integrator(nt, nr, dt, dr)
        return self.integrator

    def create_descriptive_neuron(self, background_response=0/pq.s, kernel=None,
                                  annotations={}):
        """
        Create descriptive neuron

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        kernel : function
            Impulse-response function.
        annotations : dict
            Dictionary with various annotations.

        Returns
        -------
        out : pylgn.DescriptiveNeuron
            Descriptive neuron object
        """
        neuron = DescriptiveNeuron(background_response=background_response,
                                   kernel=kernel, annotations=annotations)
        self.neurons.append(neuron)

        return neuron

    def create_ganglion_cell(self, background_response=0/pq.s, kernel=None,
                             annotations={}):
        """
        Create ganglion cell

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        kernel : function
            Impulse-response function.
        annotations : dict
            Dictionary with various annotations.

        Returns
        -------
        out : pylgn.Ganglion
            Ganglion object
        """
        ganglion = Ganglion(background_response=background_response,
                            kernel=kernel, annotations=annotations)
        self.neurons.append(ganglion)

        return ganglion

    def create_relay_cell(self, background_response=0/pq.s, annotations={}):
        """
        Create relay cell

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        annotations : dict
            Dictionary with various annotations.

        Returns
        -------
        out : pylgn.Relay
            Relay object
        """

        if len([neuron for neuron in self.neurons if type(neuron).__name__ == "Relay"]) != 0:
            raise ValueError("network already has relay cell population")

        relay = Relay(background_response, annotations=annotations)
        self.neurons.append(relay)

        return relay

    def create_cortical_cell(self, background_response=0/pq.s,
                             annotations={}):
        """
        Create cortical cell

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        annotations : dict
            Dictionary with various annotations.

        Returns
        -------
        out : pylgn.Cortical
            Cortical object
        """
        cortical = Cortical(background_response, annotations=annotations)
        self.neurons.append(cortical)

        return cortical

    def set_stimulus(self, closure, compute_fft=False):
        """
        Sets stimulus.

        Parameters
        ----------
        closure : callable (closure)
            stimulus function. If compute_fft is False the
            stimulus function should be the Fourier transform
            of the stimulus.
        compute_fft : bool
            If True numerical integration is used to calculate
            the Fourier transform of the stimulus.

        """
        self.stimulus = Stimulus(closure)
        if not compute_fft:
            w_vec, ky_vec, kx_vec = self.integrator.freq_meshgrid()
            stimulus_ft = closure(w=w_vec, kx=kx_vec, ky=ky_vec)
            self.stimulus.ft = stimulus_ft
        else:
            # print("Calculating fft of stimulus...")
            t_vec, y_vec, x_vec = self.integrator.meshgrid()
            self.stimulus.ft = self.integrator.compute_fft(closure(t=t_vec,
                                                                   x=x_vec,
                                                                   y=y_vec))

    def connect(self, source, target, kernel, weight=1.0):
        """
        Connect neurons.

        Parameters
        ----------
        source : pylgn.Neuron
            Source neuron
        target : pylgn.Neuron
            Target neuron
        kernel : function
            Connectivity kernel
        weight : float
            Connectivity weight
        """
        if isinstance(kernel, tuple):
            spatiotemporal_kernel = _unpack_kernel_tuple(kernel)
        else:
            spatiotemporal_kernel = kernel

        target.add_connection(source, spatiotemporal_kernel, weight)

    def compute_irf_ft(self, neuron):
        """
        Computes the Fourier transform of the
        impulse-response function of a neuron.

        Parameters
        ----------
        neuron : pylgn.Neuron
        """
        neuron.irf_ft_is_computed = True
        w_vec, ky_vec, kx_vec = self.integrator.freq_meshgrid()
        neuron.irf_ft = neuron.evaluate_irf_ft(w_vec, kx_vec, ky_vec)

    def compute_response_ft(self, neuron, recompute_irf_ft=False):
        """
        Computes the Fourier transform of the response of a neuron.

        Parameters
        ----------
        neuron : pylgn.Neuron
        """
        if self.stimulus.ft is None:
            raise ValueError("Stimulus is not set. Use network.set_stimulus(stimuls).")

        neuron.response_ft_is_computed = True
        if not neuron.irf_ft_is_computed or recompute_irf_ft:
            self.compute_irf_ft(neuron)

        neuron.response_ft = np.multiply(neuron.irf_ft, self.stimulus.ft)

        if neuron.background_response.magnitude != 0:
            neuron.response_ft[0, 0, 0] += 8*np.pi**3 / self.integrator.dk.magnitude**2 / self.integrator.dw.magnitude * neuron.background_response.rescale(1/pq.s).magnitude

        if self.memory_efficient:
            neuron.irf_ft_is_computed = False
            neuron.irf_ft = None
            self.stimulus.ft = None

    def compute_irf(self, neuron, recompute_ft=False):
        """
        Computes the impulse-response function of a neuron.

        Parameters
        ----------
        neuron : pylgn.Neuron
        recompute_ft : bool
            If True the Fourier transform is recalculated.
        """
        if not neuron.irf_ft_is_computed or recompute_ft:
            self.compute_irf_ft(neuron)

        neuron.irf = self.integrator.compute_inverse_fft(neuron.irf_ft) * neuron.unit

        if self.memory_efficient:
            neuron.irf_ft_is_computed = False
            neuron.irf_ft = None

    def compute_response(self, neuron, recompute_ft=False):
        """
        Computes the response of a neuron.

        Parameters
        ----------
        neuron : pylgn.Neuron
        recompute_ft : bool
            If True the Fourier transform is recalculated.
        """
        if not neuron.response_ft_is_computed or recompute_ft:
            self.compute_response_ft(neuron, recompute_irf_ft=True)

        neuron.response = self.integrator.compute_inverse_fft(neuron.response_ft) * neuron.unit

        if self.memory_efficient:
            neuron.response_ft_is_computed = False
            neuron.response_ft = None

        # NOTE half-wave rectified function for cortical cells
        if isinstance(neuron, Cortical):
            neuron.response = neuron.response.clip(min=0*neuron.unit)

    def clear(self):
        """
        Clears the neuron list.
        """
        del self.neurons[:]


##############################################################
# Stimulus
##############################################################
class Stimulus:
    """
    Stimulus class

    Attributes
    ----------
    neurons : list
         List with pylgn.Neuron objects
    integrator : pylgn.Integrator
         Integrator object
    """
    def __init__(self, closure):
        """
        Network constructor
        """
        self.spatiotemporal = None
        self.ft = None
        self.closure = closure


##############################################################
# Neuron classes
##############################################################
class Neuron(ABC):
    """
    Neuron base class.

    Attributes
    ----------
    center_response
    background_response : quantity scalar
         Background activity.
    annotations : dict
        Dictionary with various annotations on the Neuron object.
    connections : dict
        Dictionary with connected neurons including the connectivity
        kernel and weight.
    response : quantity array
         Spatiotemporal response
    response_ft : quantity array
         Fourier transformed response
    irf : quantity array
         Spatiotemporal impulse-response function
    irf_ft : quantity array
         Fourier transformed impulse-response function
    """
    def __init__(self, background_response, annotations):
        """
        Neuron constructor

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        annotations : dict
            Dictionary with various annotations on the Neuron object.
        """
        self.unit = 1. / pq.s
        self.background_response = background_response if isinstance(background_response, pq.Quantity) else background_response * self.unit
        self.connections = defaultdict(list)
        self.response = None
        self.response_ft = None
        self.irf = None
        self.irf_ft = None

        self.irf_ft_is_computed = False
        self.response_ft_is_computed = False
        self.annotations = {"background_response": background_response}
        self.annotations.update({"connections": defaultdict(list)})
        self.annotate(annotations)

    def annotate(self, annotations):
        """
        Add annotations to a Neuron object.

        Parameters
        ----------
        annotations : dict
            Dictionary containing annotations

        """
        self.annotations.update(annotations)

    def add_connection(self, neuron, kernel, weight):
        """
        Add connection to another neuron.

        Parameters
        ----------
        neuron : pylgn.Neuron
             Source neuron
        kernel : functions
                Connectivity kernel
        weight : float
              Connectivity weight
        """
        self._check_if_connection_is_allowed(neuron)
        connection = {"neuron": neuron,
                      "kernel": kernel,
                      "weight": weight}
        self.connections[type(neuron).__name__].append(connection)

        annotation = {"kernel": closure_params(kernel), "weight": weight}
        self.annotations["connections"][type(neuron).__name__.lower()].append(annotation)

    @abstractmethod
    def _check_if_connection_is_allowed(self, neuron):
        pass

    @abstractmethod
    def evaluate_irf_ft(self, w, kx, ky):
        """
        Evaluates the Fourier transform of
        impulse-response function
        """
        pass

    @property
    def center_response(self):
        """
        Response of neuron in the center of grid over time

        Returns
        -------
        out : quantity array
            Response of neuron in the center of grid over time

        """
        idx = int(self.response.shape[1] / 2)
        return np.real(self.response[:, idx, idx])


class DescriptiveNeuron(Neuron):
    def __init__(self, background_response, kernel, annotations={}):
        """
        Descriptive neuron constructor

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        kernel : function
            Impulse-response function.
        annotations : dict
            Dictionary with various annotations.
        """
        super().__init__(background_response, annotations)
        self.set_kernel(kernel)

    def _check_if_connection_is_allowed(self, neuron):
        raise TypeError("Descriptive cells cannot receive connection")

    def evaluate_irf_ft(self, w, kx, ky):
        """
        Evaluates the Fourier transform of
        impulse-response function
        """
        return self.kernel(w, kx, ky)

    def set_kernel(self, kernel):
        """
        Set the impulse-response function.

        Parameters
        ----------
        kernel : func or tuple
             Fourier transformed kernel/
             tuple of Fourier transformed spatial
             and temporal kernel
        """
        if isinstance(kernel, tuple):
            self.kernel = _unpack_kernel_tuple(kernel)
        else:
            self.kernel = kernel

        self.annotations["kernel"] = closure_params(self.kernel)


class Ganglion(Neuron):

    def __init__(self, background_response, kernel, annotations={}):
        """
        Ganglion constructor

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.
        kernel : function
            Impulse-response function.
        annotations : dict
            Dictionary with various annotations.
        """
        from .kernels.spatial import create_dog_ft
        from .kernels.temporal import create_delta_ft

        super().__init__(background_response, annotations)

        if kernel is None:
            kernel = (create_dog_ft(), create_delta_ft())
        self.set_kernel(kernel)

    def _check_if_connection_is_allowed(self, neuron):
        raise TypeError("Ganglion cells cannot receive connection")

    def evaluate_irf_ft(self, w, kx, ky):
        """
        Evaluates the Fourier transform of
        impulse-response function
        """
        return self.kernel(w, kx, ky)

    def set_kernel(self, kernel):
        """
        Set the impulse-response function.

        Parameters
        ----------
        kernel : func or tuple
             Fourier transformed kernel/
             tuple of Fourier transformed spatial
             and temporal kernel
        """
        if isinstance(kernel, tuple):
            self.kernel = _unpack_kernel_tuple(kernel)
        else:
            self.kernel = kernel

        self.annotations["kernel"] = closure_params(self.kernel)


class Relay(Neuron):
    def __init__(self, background_response, annotations={}):
        """
        Relay constructor

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.

        annotations : dict
            Dictionary with various annotations.
        """
        super().__init__(background_response, annotations)

    def _check_if_connection_is_allowed(self, neuron):
        if not isinstance(neuron, (Ganglion, Cortical)):
            raise TypeError("Unsupported connection to relay cell: ", type(neuron))

    def evaluate_irf_ft(self, w, kx, ky):
        """
        Evaluates the Fourier transform of
        impulse-response function
        """
        retinal_input = 0
        cortical_input = 0

        for c in self.connections["Ganglion"]:
            Krg = c["kernel"](w, kx, ky)
            Wg = c["neuron"].evaluate_irf_ft(w, kx, ky)
            w_rg = c["weight"]
            retinal_input += w_rg * Krg * Wg

        for c in self.connections["Cortical"]:
            thalamocortical_connection = c["neuron"].connections["Relay"][0]
            Kcr = thalamocortical_connection["kernel"](w, kx, ky)
            Krc = c["kernel"](w, kx, ky)
            w_rc = c["weight"]
            w_cr = thalamocortical_connection["weight"]
            cortical_input += w_rc * w_cr * Krc * Kcr

        return retinal_input / (1 - cortical_input)


class Cortical(Neuron):
    def __init__(self, background_response, annotations={}):
        """
        Cortical constructor

        Parameters
        ----------
        background_response : quantity scalar
            Background activity.

        annotations : dict
            Dictionary with various annotations.
        """
        super().__init__(background_response, annotations)

    def evaluate_irf_ft(self, w, kx, ky):
        """
        Evaluates the Fourier transform of
        impulse-response function
        """
        thalamic_input = 0
        for c in self.connections["Relay"]:
            Kcr = c["kernel"](w, kx, ky)
            Rr = c["neuron"].evaluate_irf_ft(w, kx, ky)
            w_cr = c["weight"]
            thalamic_input += w_cr * Kcr * Rr

        return thalamic_input

    def _check_if_connection_is_allowed(self, neuron):
        if not isinstance(neuron, Relay):
            raise TypeError("Unsupported connection to cortical cell: ", type(neuron))
        if type(neuron).__name__ in self.connections:
            raise ValueError("Cortical cell already has a thalamic connection: ",
                             self.connections)
