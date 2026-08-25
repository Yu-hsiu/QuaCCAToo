"""
This module provides functions to save and load quaccatoo objects, such as instances from QSys,
ExpData, Analysis and PulsedSim.
It also contains small internal helpers shared by the plotting methods of the other modules.
"""

import os
import shutil
import warnings
import zipfile

import dill
import numpy as np
from qutip import Qobj, fileio

__all__ = ["load_quaccatoo", "save_quaccatoo"]

####################################################################################################


def _check_figsize(figsize) -> None:
    """
    Internal helper checking that figsize is a tuple of two positive numbers,
    as expected by matplotlib.pyplot.

    Parameters
    ----------
    figsize : tuple
        Size of the figure to be passed to matplotlib.pyplot

    Raises
    ------
    ValueError
        If figsize is not a tuple of two positive numbers.
    """
    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or not all(isinstance(dim, (int, float)) and dim > 0 for dim in figsize)
    ):
        raise ValueError(f"figsize must be a tuple of two positive floats, got {figsize}.")


# Below this value a solver tolerance is at the level of the double precision epsilon that the
# integrator accumulates anyway over the internal steps of a long sequence. Asking for it costs
# a large factor in runtime without making the result more accurate.
_MIN_USEFUL_TOL = 1e-14


def _warn_extreme_tolerances(options: dict) -> None:
    """
    Warn when the solver tolerances are set below what double precision can deliver.

    A dynamical decoupling sequence integrates thousands of internal steps per pulse, so the
    accumulated round-off is orders of magnitude above 1e-16. Requesting such a tolerance makes
    the integrator take many more steps for a result that does not change, which turns long
    sequences into runs of several hours.

    Parameters
    ----------
    options : dict
        Dictionary of solver options to be passed to QuTip.
    """
    extreme = {
        key: options[key]
        for key in ("atol", "rtol")
        if isinstance(options.get(key), (int, float)) and options[key] < _MIN_USEFUL_TOL
    }

    if extreme:
        warnings.warn(
            f"{extreme} is below the double precision floor of {_MIN_USEFUL_TOL:.0e} that the "
            "integrator can hold over a long sequence. The extra steps multiply the runtime "
            "without changing the result.",
            stacklevel=3,
        )


def _usable_cpus() -> int:
    """
    Return the number of cores the current process is actually allowed to run on.

    os.cpu_count reports the cores of the machine, which on containerized runtimes is larger
    than the quota given to the container. sched_getaffinity reflects the quota and exists on
    Linux only, so it is used when available.

    Returns
    -------
    int
        Number of usable cores, at least one.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def save_quaccatoo(obj_save, file_path):
    """
    Look for all the attributes of the obj, save Qobj attributes to separate files,
    and save the rest of the attributes to a pickle file. Finally, create a zip file
    containing all the files.

    Parameters
    ----------
    obj_save : quaccatoo obj
        The obj to be saved.
    file_path : str
        Path to the file where the attributes will be saved.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")  # noqa: TRY004

    tmp_dir = "tmp"

    try:
        # create a temporary directory to store files and get a list of all the attributes defined
        # in the obj
        os.makedirs(tmp_dir, exist_ok=True)
        attributes = list(obj_save.__dict__.keys())
        py_attr = []

        # Separate attributes into Python and Qobj types
        for attr in attributes:
            value = getattr(obj_save, attr)
            if isinstance(value, Qobj) or (
                isinstance(value, (list, np.ndarray))
                and len(value) > 0
                and all(isinstance(item, Qobj) for item in value)
            ):
                # Save Qobj attributes to a file in the temporary directory
                fileio.qsave(value, os.path.join(tmp_dir, str(attr)))
            else:
                py_attr.append(attr)

        # Create a dictionary to store the python attributes names and values
        py_data = {
            "__type__": obj_save.__class__.__name__,
            **{attr: getattr(obj_save, attr) for attr in py_attr},
        }

        # Save the python data to a file in the temporary directory
        with open(os.path.join(tmp_dir, "py_data.pkl"), "wb") as f_pkl:
            dill.dump(py_data, f_pkl)

        # Create a zip file to store all the files
        with zipfile.ZipFile(file_path, "w") as zip_file:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, start=tmp_dir)
                    zip_file.write(file_path, rel_path)

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def load_quaccatoo(file_path):
    """
    Loads the attributes of an obj from a zip file,
    creates an instance of the obj, and sets the attributes.

    Parameters
    ----------
    file_path : str
        Path to the zip file where the attributes are saved.

    Returns
    -------
    obj
        The loaded obj.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_name must be a string")  # noqa: TRY004

    tmp_dir = "tmp"

    import quaccatoo  # pylint: disable=import-outside-toplevel,cyclic-import

    try:
        # Extract the zip file to a temporary directory
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(tmp_dir)

        # Load the py_data.pkl file
        with open(os.path.join(tmp_dir, "py_data.pkl"), "rb") as f_pkl:
            py_data = dill.load(f_pkl)

        # Get the obj name from the py_data and create new instance
        cls = getattr(quaccatoo, py_data["__type__"])
        obj_load = cls.__new__(cls)

        # Load the attributes from the py_data
        for attr, value in py_data.items():
            if attr != "__type__":
                setattr(obj_load, attr, value)

        # Load the Qobj attributes from the files
        for file in os.listdir(tmp_dir):
            if file != "py_data.pkl":
                attr_name = os.path.splitext(file)[0]
                value = fileio.qload(os.path.join(tmp_dir, attr_name))
                setattr(obj_load, attr_name, value)

        return obj_load

    finally:
        if os.path.exists(tmp_dir):
            # Remove the temporary directory
            shutil.rmtree(tmp_dir)
