"""
Wrappers that provide a minimal interface to kimpy methods and objects

Daniel S. Karls
University of Minnesota
"""

import functools

import numpy as np
import kimpy

from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError

# Function used for casting parameter/extent indices to C-compatible ints
c_int = np.intc

# Function used for casting floating point parameter values to C-compatible
# doubles
c_double = np.double


def c_int_args(func):
    """
    Decorator for instance methods that will cast all of the args passed,
    excluding the first (which corresponds to 'self'), to C-compatible
    integers.
    """

    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        args_cast = [args[0]]
        args_cast += map(c_int, args[1:])
        return func(*args, **kwargs)

    return myfunc


def check_call(f, *args):
    """
    Given a function that returns either an integer error code or a
    tuple whose last element is an integer error code, call it with the
    requested arguments.  If a non-zero error code was returned, raise
    an exception.  Otherwise, pass along the rest of the objects
    returned by the function call.
    """

    def _check_error(error, msg):
        if error != 0 and error is not None:
            raise KimpyError('Calling "{}" failed.'.format(msg))

    ret = f(*args)

    if isinstance(ret, int):
        # Only an error code was returned
        _check_error(ret, f.__name__)
    else:
        # An error code plus other variables were returned
        error = ret[-1]
        _check_error(error, f.__name__)

        if len(ret[:-1]) == 1:
            # Pick the single remaining element out of the tuple
            return ret[0]
        else:
            # Return the tuple containing the rest of the elements
            return ret[:-1]


def check_call_wrapper(func):
    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        return check_call(func, *args)

    return myfunc


# kimpy methods wrapped in ``check_error``
collections_create = functools.partial(check_call, kimpy.collections.create)
model_create = functools.partial(check_call, kimpy.model.create)
simulator_model_create = functools.partial(check_call, kimpy.simulator_model.create)
get_species_name = functools.partial(check_call, kimpy.species_name.get_species_name)

# kimpy attributes (here to avoid importing kimpy in higher-level modules)
collection_item_type_portableModel = kimpy.collection_item_type.portableModel


class ModelCollections:
    """
    KIM Portable Models and Simulator Models are installed/managed into
    different "collections".  In order to search through the different
    KIM API model collections on the system, a corresponding object must
    be instantiated.  For more on model collections, see the KIM API's
    install file:
    https://github.com/openkim/kim-api/blob/master/INSTALL
    """

    def __init__(self):
        self.collection = collections_create()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.destroy()

    def get_item_type(self, model_name):
        try:
            model_type = check_call(self.collection.get_item_type, model_name)
        except KimpyError:
            msg = (
                "Could not find model {} installed in any of the KIM API model "
                "collections on this system.  See "
                "https://openkim.org/doc/usage/obtaining-models/ for instructions on "
                "installing models.".format(model_name)
            )
            raise KIMModelNotFound(msg)

        return model_type

    def destroy(self):
        if self.initialized:
            kimpy.collections.destroy(self.collection)
            del self.collection

    @property
    def initialized(self):
        return hasattr(self, "collection")


class PortableModel:
    """ Creates a KIM API Portable Model object and provides a minimal interface to it
    """

    def __init__(self, model_name, debug):
        self.model_name = model_name
        self.debug = debug

        # Create KIM API Model object
        units_accepted, self.kim_model = model_create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.model_name,
        )

        if not units_accepted:
            raise KIMModelInitializationError(
                "Requested units not accepted in kimpy.model.create"
            )

        if self.debug:
            l_unit, e_unit, c_unit, te_unit, ti_unit = self.kim_model.get_units()
            print("Length unit is: {}".format(l_unit))
            print("Energy unit is: {}".format(e_unit))
            print("Charge unit is: {}".format(c_unit))
            print("Temperature unit is: {}".format(te_unit))
            print("Time unit is: {}".format(ti_unit))
            print()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.destroy()

    def get_model_supported_species_and_codes(self):
        """Get all of the supported species for this model and their
        corresponding integer codes that are defined in the KIM API

        Returns
        -------
        species : list of str
            Abbreviated chemical symbols of all species the mmodel
            supports (e.g. ["Mo", "S"])

        codes : list of int
            Integer codes used by the model for each species (order
            corresponds to the order of ``species``)
        """
        species = []
        codes = []
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name = get_species_name(i)
            species_support, code = self.get_species_support_and_code(species_name)

            if species_support:
                species.append(str(species_name))
                codes.append(code)

        return species, codes

    def clear_then_refresh(self):
        return self.kim_model.clear_then_refresh()

    def _get_number_of_parameters(self):
        return self.kim_model.get_number_of_parameters()

    @c_int_args
    def _get_parameter_metadata(self, index_parameter):
        return self.kim_model.get_parameter_metadata(index_parameter)

    @c_int_args
    def _get_parameter_int(self, index_param, index_extent):
        return self.kim_model.get_parameter_int(index_param, index_extent)

    @c_int_args
    def _get_parameter_double(self, index_param, index_extent):
        return self.kim_model.get_parameter_double(index_param, index_extent)

    def _set_parameter(self, index_param, index_extent, value_typecast):
        return self.kim_model.set_parameter(
            c_int(index_param), c_int(index_extent), value_typecast
        )

    def parameters_metadata(self):
        """Metadata associated with all model parameters.

        Returns
        -------
        dict
            Meta data associated with all model parameters.
        """
        num_params = self._get_number_of_parameters()
        metadata = {}
        for ii in range(num_params):
            metadata.update(self._get_one_parameter_metadata(ii))
        return metadata

    def parameter_names(self):
        """Names of model parameters registered in the KIM API.

        Returns
        -------
        list
            Names of model parameters registered in the KIM API
        """
        nparams = self._get_number_of_parameters()
        names = []
        for ii in range(nparams):
            name = list(self._get_one_parameter_metadata(ii))[0]
            names.append(name)
        return names

    def get_parameters(self, **kwargs):
        """
        Get the values of one or more model parameter arrays.

        Given the names of one or more model parameters and a set of indices
        for each of them, retrieve the corresponding elements of the relevant
        model parameter arrays.

        Parameters
        ----------
        **kwargs
            Names of the model parameters and the indices whose values should
            be retrieved.

        Returns
        -------
        dict
            The requested indices and the values of the model's parameters.

        Note
        ----
        The output of this method can be used as input of
        ``set_parameters``.

        Example
        -------
        To get `epsilons` and `sigmas` in the LJ universal model for Mo-Mo
        (index 4879), Mo-S (index 2006) and S-S (index 1980) interactions::

            >>> LJ = 'LJ_ElliottAkerson_2015_Universal__MO_959249795837_003'
            >>> calc = KIM(LJ)
            >>> calc.get_parameters(epsilons=[4879, 2006, 1980],
            ...                     sigmas=[4879, 2006, 1980])
            {'epsilons': [[4879, 2006, 1980],
                          [4.47499, 4.421814057295943, 4.36927]],
             'sigmas': [[4879, 2006, 1980],
                        [2.74397, 2.30743, 1.87089]]}
        """
        parameters = {}
        for parameter_name, index_range in kwargs.items():
            parameters.update(self._get_one_parameter(parameter_name, index_range))
        return parameters

    def set_parameters(self, **kwargs):
        """
        Set the values of one or more model parameter arrays.

        Given the names of one or more model parameters and a set of indices
        and corresponding values for each of them, mutate the corresponding
        elements of the relevant model parameter arrays.

        Parameters
        ----------
        **kwargs
            Names of the model parameters to mutate and the corresponding
            indices and values to set.

        Returns
        -------
        dict
            The requested indices and the values of the model's parameters
            that were set.

        Example
        -------
        To set `epsilons` in the LJ universal model for Mo-Mo (index 4879),
        Mo-S (index 2006) and S-S (index 1980) interactions to 5.0, 4.5, and
        4.0, respectively::

            >>> LJ = 'LJ_ElliottAkerson_2015_Universal__MO_959249795837_003'
            >>> calc = KIM(LJ)
            >>> calc.set_parameters(epsilons=[[4879, 2006, 1980],
            ...                               [5.0, 4.5, 4.0]])
            {'epsilons': [[4879, 2006, 1980],
                          [5.0, 4.5, 4.0]]}
        """
        parameters = {}
        for parameter_name, parameter_data in kwargs.items():
            self._set_one_parameter(
                parameter_name, parameter_data[0], parameter_data[1]
            )
            parameters.update({parameter_name: parameter_data})

        return parameters

    def _get_one_parameter(self, parameter_name, index_range):
        """
        Retrieve the value of one or more components of a model parameter array.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.
        index_range : int or list
            Zero-based index (int) or indices (list of int) specifying the
            component(s) of the corresponding model parameter array that are
            to be retrieved.

        Returns
        -------
        dict
            The requested indices and the corresponding values of the model
            parameter array.
        """
        (parameter_name_index, dtype, index_range_dim) = self._one_parameter_data(
            parameter_name, index_range
        )

        if index_range_dim == 0:
            values = self._get_one_value(parameter_name_index, index_range, dtype)
        elif index_range_dim == 1:
            values = []
            for idx in index_range:
                values.append(self._get_one_value(parameter_name_index, idx, dtype))
        else:
            raise ValueError("Index range must be an integer or a list of integer")
        return {parameter_name: [index_range, values]}

    def _set_one_parameter(self, parameter_name, index_range, values):
        """
        Set the value of one or more components of a model parameter array.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.
        index_range : int or list
            Zero-based index (int) or indices (list of int) specifying the
            component(s) of the corresponding model parameter array that are
            to be mutated.
        values : int/float or list
            Value(s) to assign to the component(s) of the model parameter
            array specified by ``index_range``.
        """
        (parameter_name_index, dtype, index_range_dim) = self._one_parameter_data(
            parameter_name, index_range
        )
        values_dim = np.ndim(values)

        # Check the shape of index_range and values
        msg = "index_range and values must have the same shape"
        assert index_range_dim == values_dim, msg

        if index_range_dim == 0:
            self._set_one_value(parameter_name_index, index_range, dtype, values)
        elif index_range_dim == 1:
            assert len(index_range) == len(values), msg
            for idx, value in zip(index_range, values):
                self._set_one_value(parameter_name_index, idx, dtype, value)
        else:
            raise ValueError(
                "Index range must be an integer or a list containing a single integer"
            )

    def _get_one_parameter_metadata(self, index_parameter):
        """
        Get metadata associated with a single model parameter.

        Parameters
        ----------
        index_parameter : int
            Zero-based index used by the KIM API to refer to this model
            parameter.

        Returns
        -------
        dict
            Metadata associated with the requested model parameter.
        """
        out = self._get_parameter_metadata(index_parameter)
        dtype, extent, name, description, error = out
        dtype = repr(dtype)
        pdata = {
            name: {
                "dtype": dtype,
                "extent": extent,
                "description": description,
                "error": error,
            }
        }
        return pdata

    def _get_parameter_name_index(self, parameter_name):
        """
        Given the name of a model parameter, find the index used by the KIM
        API to refer to it.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.

        Returns
        -------
        int
            Zero-based index used by the KIM API to refer to this model parameter.
        """
        parameter_name_index = np.where(
            np.asarray(self.parameter_names()) == parameter_name
        )[0]
        return parameter_name_index

    def _get_one_value(self, index_param, index_extent, dtype):
        """
        Retrieve the value of a single component of a model parameter array.

        Parameters
        ----------
        index_param : int
            Zero-based index used by the KIM API to refer to this model
            parameter.
        index_extent : int
            Zero-based index locating the component of the model parameter
            array that is to be retrieved.
        dtype : "Integer" or "Double"
            Data type of the model's parameter.  Allowed types: "Integer" or
            "Double".

        Returns
        -------
        int or float
            Value of the requested component of the model parameter array.

        Raises
        ------
        ValueError
            If ``dtype`` is not one of "Integer" or "Double"
        """
        self._check_parameter_data_type(dtype)

        if dtype == "Double":
            pp = self._get_parameter_double(index_param, index_extent)[0]
        elif dtype == "Integer":
            pp = self._get_parameter_int(index_param, index_extent)[0]

        return pp

    def _set_one_value(self, index_param, index_extent, dtype, value):
        """
        Set the value of a single component of a model parameter array.

        Parameters
        ----------
        index_param : int
            Zero-based index used by the KIM API to refer to this model
            parameter.
        index_extent : int
            Zero-based index locating the component of the model parameter
            array that is to be mutated.
        dtype : "Integer" or "Double"
            Data type of the model's parameter.  Allowed types: "Integer" or
            "Double".
        value : int or float
            Value to assign the the requested component of the model
            parameter array.

        Raises
        ------
        ValueError
            If ``dtype`` is not one of "Integer" or "Double".
        """
        self._check_parameter_data_type(dtype)

        if dtype == "Double":
            value_typecast = c_double(value)
        elif dtype == "Integer":
            value_typecast = c_int(value)

        self._set_parameter(index_param, index_extent, value_typecast)

    def _one_parameter_data(self, parameter_name, index_range):
        """Get the data of one of the parameter. The data will be used in
        ``_get_one_parameter`` and ``_set_one_parameter``.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.:w

        index_range : int or list
            Zero-based index (int) or indices (list of int) specifying the
            to be mutated.
            component(s) of the corresponding model parameter array that are

        Returns
        -------
        list
            Contains index of model's parameter, metadata of the parameter,
            dtype, and dimension of ``index_range``.

        Raises
        ------
        ValueError
            If ``parameter_name`` is not registered in the KIM API.
        """
        # Check if model has parameter_name
        if parameter_name not in self.parameter_names():
            raise ValueError(f"Parameter {parameter_name} is not supported.")

        parameter_name_index = self._get_parameter_name_index(parameter_name)
        metadata = self._get_one_parameter_metadata(parameter_name_index)
        dtype = list(metadata.values())[0]["dtype"]

        index_range_dim = np.ndim(index_range)

        return parameter_name_index, dtype, index_range_dim

    @staticmethod
    def _check_parameter_data_type(dtype):
        if dtype not in ["Integer", "Double"]:
            raise ValueError(
                f"Invalid data type {dtype}.  Allowed values are "
                "'Integer' or 'Double'."
            )

    @check_call_wrapper
    def compute(self, compute_args_wrapped, release_GIL):
        return self.kim_model.compute(compute_args_wrapped.compute_args, release_GIL)

    @check_call_wrapper
    def get_species_support_and_code(self, species_name):
        return self.kim_model.get_species_support_and_code(species_name)

    def get_influence_distance(self):
        return self.kim_model.get_influence_distance()

    def get_neighbor_list_cutoffs_and_hints(self):
        return self.kim_model.get_neighbor_list_cutoffs_and_hints()

    def compute_arguments_create(self):
        return ComputeArguments(self, self.debug)

    def compute_arguments_destroy(self, compute_args_wrapped):
        compute_args_wrapped.destroy()

    def destroy(self):
        if self.initialized:
            kimpy.model.destroy(self.kim_model)
            del self.kim_model

    @property
    def initialized(self):
        return hasattr(self, "kim_model")


class ComputeArguments:
    """
    Creates a KIM API ComputeArguments object from a KIM Portable Model object and
    configures it for ASE.  A ComputeArguments object is associated with a KIM Portable
    Model and is used to inform the KIM API of what the model can compute.  It is also
    used to register the data arrays that allow the KIM API to pass the atomic
    coordinates to the model and retrieve the corresponding energy and forces, etc.
    """

    def __init__(self, kim_model_wrapped, debug):
        self.kim_model_wrapped = kim_model_wrapped
        self.debug = debug

        # Create KIM API ComputeArguments object
        self.compute_args = check_call(
            self.kim_model_wrapped.kim_model.compute_arguments_create
        )

        # Check compute arguments
        kimpy_arg_name = kimpy.compute_argument_name
        num_arguments = kimpy_arg_name.get_number_of_compute_argument_names()
        if self.debug:
            print("Number of compute_args: {}".format(num_arguments))

        for i in range(num_arguments):
            name = check_call(kimpy_arg_name.get_compute_argument_name, i)
            dtype = check_call(kimpy_arg_name.get_compute_argument_data_type, name)

            arg_support = self.get_argument_support_status(name)

            if self.debug:
                print(
                    "Compute Argument name {:21} is of type {:7} and has support "
                    "status {}".format(*[str(x) for x in [name, dtype, arg_support]])
                )

            # See if the model demands that we ask it for anything other than energy and
            # forces.  If so, raise an exception.
            if arg_support == kimpy.support_status.required:
                if (
                    name != kimpy.compute_argument_name.partialEnergy
                    and name != kimpy.compute_argument_name.partialForces
                ):
                    raise KIMModelInitializationError(
                        "Unsupported required ComputeArgument {}".format(name)
                    )

        # Check compute callbacks
        callback_name = kimpy.compute_callback_name
        num_callbacks = callback_name.get_number_of_compute_callback_names()
        if self.debug:
            print()
            print("Number of callbacks: {}".format(num_callbacks))

        for i in range(num_callbacks):
            name = check_call(callback_name.get_compute_callback_name, i)

            support_status = self.get_callback_support_status(name)

            if self.debug:
                print(
                    "Compute callback {:17} has support status {}".format(
                        str(name), support_status
                    )
                )

            # Cannot handle any "required" callbacks
            if support_status == kimpy.support_status.required:
                raise KIMModelInitializationError(
                    "Unsupported required ComputeCallback: {}".format(name)
                )

    @check_call_wrapper
    def set_argument_pointer(self, compute_arg_name, data_object):
        return self.compute_args.set_argument_pointer(compute_arg_name, data_object)

    @check_call_wrapper
    def get_argument_support_status(self, name):
        return self.compute_args.get_argument_support_status(name)

    @check_call_wrapper
    def get_callback_support_status(self, name):
        return self.compute_args.get_callback_support_status(name)

    @check_call_wrapper
    def set_callback(self, compute_callback_name, callback_function, data_object):
        return self.compute_args.set_callback(
            compute_callback_name, callback_function, data_object
        )

    @check_call_wrapper
    def set_callback_pointer(self, compute_callback_name, callback, data_object):
        return self.compute_args.set_callback_pointer(
            compute_callback_name, callback, data_object
        )

    def update(
        self, num_particles, species_code, particle_contributing, coords, energy, forces
    ):
        """ Register model input and output in the kim_model object."""
        compute_arg_name = kimpy.compute_argument_name
        set_argument_pointer = self.set_argument_pointer

        set_argument_pointer(compute_arg_name.numberOfParticles, num_particles)
        set_argument_pointer(compute_arg_name.particleSpeciesCodes, species_code)
        set_argument_pointer(
            compute_arg_name.particleContributing, particle_contributing
        )
        set_argument_pointer(compute_arg_name.coordinates, coords)
        set_argument_pointer(compute_arg_name.partialEnergy, energy)
        set_argument_pointer(compute_arg_name.partialForces, forces)

        if self.debug:
            print("Debug: called update_kim")
            print()

    @check_call_wrapper
    def destroy(self):
        return self.kim_model_wrapped.kim_model.compute_arguments_destroy(
            self.compute_args
        )


class SimulatorModel:
    """ Creates a KIM API Simulator Model object and provides a minimal
    interface to it.  This is only necessary in this package in order to
    extract any information about a given simulator model because it is
    generally embedded in a shared object.
    """

    def __init__(self, model_name):
        # Create a KIM API Simulator Model object for this model
        self.model_name = model_name
        self.simulator_model = simulator_model_create(self.model_name)

        # Need to close template map in order to access simulator model metadata
        self.simulator_model.close_template_map()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.destroy()

    def destroy(self):
        if self.initialized:
            kimpy.simulator_model.destroy(self.simulator_model)
            del self.simulator_model

    @property
    def simulator_name(self):
        simulator_name, _ = self.simulator_model.get_simulator_name_and_version()
        return simulator_name

    @property
    def num_supported_species(self):
        num_supported_species = self.simulator_model.get_number_of_supported_species()
        if num_supported_species == 0:
            raise KIMModelInitializationError(
                "Unable to determine supported species of simulator model {}.".format(
                    self.model_name
                )
            )
        else:
            return num_supported_species

    @property
    def supported_species(self):
        supported_species = []
        for spec_code in range(self.num_supported_species):
            species = check_call(self.simulator_model.get_supported_species, spec_code)
            supported_species.append(species)

        return tuple(supported_species)

    @property
    def num_metadata_fields(self):
        return self.simulator_model.get_number_of_simulator_fields()

    @property
    def metadata(self):
        sm_metadata_fields = {}
        for field in range(self.num_metadata_fields):
            extent, field_name = check_call(
                self.simulator_model.get_simulator_field_metadata, field
            )
            sm_metadata_fields[field_name] = []
            for ln in range(extent):
                field_line = check_call(
                    self.simulator_model.get_simulator_field_line, field, ln
                )
                sm_metadata_fields[field_name].append(field_line)

        return sm_metadata_fields

    @property
    def supported_units(self):
        try:
            supported_units = self.metadata["units"][0]
        except (KeyError, IndexError):
            raise KIMModelInitializationError(
                "Unable to determine supported units of simulator model {}.".format(
                    self.model_name
                )
            )

        return supported_units

    @property
    def atom_style(self):
        """
        See if a 'model-init' field exists in the SM metadata and, if
        so, whether it contains any entries including an "atom_style"
        command.  This is specific to LAMMPS SMs and is only required
        for using the LAMMPSrun calculator because it uses
        lammps.inputwriter to create a data file.  All other content in
        'model-init', if it exists, is ignored.
        """
        atom_style = None
        for ln in self.metadata.get("model-init", []):
            if ln.find("atom_style") != -1:
                atom_style = ln.split()[1]

        return atom_style

    @property
    def model_defn(self):
        return self.metadata["model-defn"]

    @property
    def initialized(self):
        return hasattr(self, "simulator_model")
