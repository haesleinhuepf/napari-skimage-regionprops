
import napari
from toolz import curry
from typing import Callable
from functools import wraps
import time
import inspect
import numpy as np
import pandas as pd
# most imports here are just for backwards compatbility
#from ._workflow import WorkflowManager, CURRENT_TIME_FRAME_DATA, _get_layer_from_data, _break_down_4d_to_2d_kwargs, _viewer_has_layer


@curry
def analyze_all_frames(function: Callable) -> Callable:
    from napari_workflows._workflow import _get_layer_from_data

    @wraps(function)
    def worker_function(*args, **kwargs):
        args = list(args)
        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        # Retrieve the viewer parameter so that we can know which current timepoint is selected
        viewer = None
        for key, value in bound.arguments.items():
            if isinstance(value, napari.Viewer):
                viewer = value
                viewer_key = key

        labels_layer = None
        original_args = copy_dict(bound.arguments)

        if viewer is not None:
            variable_timepoint = list(viewer.dims.current_step)
            current_timepoint = variable_timepoint[0]
            max_time = int(viewer.dims.range[-4][1])
        else:
            max_time = 0
            for key, value in original_args.items():
                if isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                                         "<class 'dask.array.core.Array'>"]:
                    if len(value.shape) == 4 and max_time < value.shape[0]:
                        max_time = value.shape[0]

        original_args = copy_dict(bound.arguments)

        result = None

        for f in range(max_time):
            print("analyzing frame", f)
            args = copy_dict(original_args)

            if viewer is None:
                for key, value in args.items():
                    if isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                                             "<class 'dask.array.core.Array'>"]:
                        if len(value.shape) == 4:
                            new_value = value[f]
                            if new_value.shape[0] == 1:
                                new_value = new_value[0]
                            args[key] = new_value
                        elif len(value.shape) == 3:
                            # keep a 3D label image for example
                            pass
                        else:
                            raise NotImplementedError("Analyzing all frames only supports combination of 3D and 4D-data")

            else:
                # in case of 4D-data (timelapse) crop out the current 3D timepoint
                if len(viewer.dims.current_step) != 4:
                    raise NotImplementedError("Analyzing all frames only supports 4D-data")

                variable_timepoint[0] = f
                viewer.dims.current_step = variable_timepoint

                from napari_workflows._workflow import _break_down_4d_to_2d_kwargs
                _break_down_4d_to_2d_kwargs(args, f, viewer)
                args[viewer_key] = None
            bound.arguments = args

            # call the decorated function
            result_single_frame = function(*bound.args, **bound.kwargs)

            result_single_frame['frame'] = [f] * len(result_single_frame['label'])
            if result is None:
                result = pd.DataFrame(result_single_frame)
            else:
                result = pd.concat([result, pd.DataFrame(result_single_frame)], ignore_index=True)

        if viewer is not None:
            # reset viewer
            variable_timepoint[0] = current_timepoint
            viewer.dims.current_step = variable_timepoint

            # find a labels layer to attach result
            for key, value in original_args.items():
                if isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                                         "<class 'dask.array.core.Array'>"]:
                    layer = _get_layer_from_data(viewer, value)
                    if isinstance(layer, napari.layers.Labels):
                        labels_layer = layer

            if labels_layer is not None:
                labels_layer.properties = result.to_dict(orient='list')

                from ._table import add_table
                add_table(labels_layer, viewer)
        else:
            return result.to_dict()

    return worker_function


def copy_dict(source, result=None):
    if result is None:
        result = {}

    for k, v in source.items():
        result[k] = v
    return result
