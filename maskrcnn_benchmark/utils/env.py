# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys

from maskrcnn_benchmark.utils.imports import import_file


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file that performs
    custom setup work that may be necessary to their computing environment.
    """
    custom_module_path = os.environ.get("TORCH_DETECTRON_ENV_MODULE")
    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module_path):
    """Load custom environment setup from a Python source file and run the setup
    function.
    """
    module = import_file("maskrcnn_benchmark.utils.env.custom_module", custom_module_path)
    assert hasattr(module, "setup_environment") and callable(
        module.setup_environment
    ), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(
        custom_module_path
    )
    module.setup_environment()


# Force environment setup when this module is imported
setup_environment()

def get_runtime_dir():
    """Retrieve the path to the runtime directory."""
    return os.getcwd()


def get_py_bin_ext():
    """Retrieve python binary extension."""
    return '.py'


def set_up_matplotlib():
    """Set matplotlib up."""
    import matplotlib
    # Use a non-interactive backend
    matplotlib.use('Agg')


def exit_on_error():
    """Exit from a detectron tool when there's an error."""
    sys.exit(1)