"""Load a YOLO model preferring its TensorRT engine when this machine can run it.

The configs always name the .pt (e.g. ``model_file: yolo26n-seg.pt``). With
``use_tensorrt: auto`` the node uses ``<same name>.engine`` from the same folder when
CUDA is available and the engine loads and runs here, and falls back to the .pt otherwise.
So the same config works on the Jetson (engine) and on the NUC (.pt).

An engine only runs on the GPU / TensorRT version it was built on, so it must be built on
each machine: ``ros2 run fbot_recognition export_engines``.
"""
import os

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor

USE_TENSORRT_PARAM = 'use_tensorrt'
USE_TENSORRT_VALUES = ('auto', 'true', 'false')


def declare_use_tensorrt(node, default='auto'):
    """Declare `use_tensorrt` accepting both strings and YAML booleans (true/false)."""
    node.declare_parameter(
        USE_TENSORRT_PARAM, default,
        ParameterDescriptor(
            dynamic_typing=True,
            description="TensorRT engine: 'auto' (use it if possible, else .pt), 'true' (required) or 'false' (.pt only)"
        )
    )


def read_use_tensorrt(node):
    value = str(node.get_parameter(USE_TENSORRT_PARAM).value).strip().lower()
    if value not in USE_TENSORRT_VALUES:
        node.get_logger().warn(f"Invalid {USE_TENSORRT_PARAM}: '{value}', using 'auto'")
        value = 'auto'
    return value


def engine_path_for(model_path):
    return os.path.splitext(model_path)[0] + '.engine'


def load_yolo_model(model_path, use_tensorrt='auto', logger=None):
    """Return (YOLO model, path actually loaded)."""
    import torch
    from ultralytics import YOLO

    log_info = logger.info if logger else print
    log_warn = logger.warn if logger else print

    engine_path = engine_path_for(model_path)

    if use_tensorrt != 'false' and not model_path.endswith('.engine'):
        reason = None
        if not torch.cuda.is_available():
            reason = 'CUDA is not available'
        elif not os.path.exists(engine_path):
            reason = f'{os.path.basename(engine_path)} not found (build it with: ros2 run fbot_recognition export_engines)'
        else:
            try:
                model = YOLO(engine_path)
                # warm-up: an engine built on another GPU / TensorRT version fails here
                model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                log_info(f'Using TensorRT engine: {engine_path}')
                return model, engine_path
            except Exception as e:
                reason = f'could not run {os.path.basename(engine_path)}: {e}'

        if use_tensorrt == 'true':
            raise RuntimeError(f'use_tensorrt is true but the TensorRT engine is unusable: {reason}')
        log_warn(f'Not using TensorRT ({reason}); loading {os.path.basename(model_path)}')

    model = YOLO(model_path)
    log_info(f'Using model: {model_path}')
    return model, model_path
