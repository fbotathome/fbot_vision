"""Build TensorRT engines (.engine) for the YOLO models of fbot_recognition, on this machine.

An engine only runs on the GPU / TensorRT version it was built on, so run this on each
machine that should use TensorRT (e.g. the Jetson), and again after a JetPack/TensorRT
upgrade. Building takes a few minutes per model.

    ros2 run fbot_recognition export_engines                     # models named in config/*.yaml
    ros2 run fbot_recognition export_engines yolo26n-seg.pt      # specific models
    ros2 run fbot_recognition export_engines --force             # rebuild existing engines

The engines are written next to the .pt in the installed weights folder, where the nodes
look for them (use_tensorrt: auto). Copy them to src/.../fbot_recognition/weights too if
the install folder may be wiped (.engine files are git-ignored).
"""
import argparse
import glob
import os
import sys
import time

import yaml
from ament_index_python.packages import get_package_share_directory


def models_from_configs(config_dir):
    """model_file (.pt) values found in the node configs."""
    found = set()

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'model_file' and isinstance(v, str) and v.endswith('.pt'):
                    found.add(v)
                else:
                    visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for path in glob.glob(os.path.join(config_dir, '*.yaml')):
        try:
            with open(path) as f:
                visit(yaml.safe_load(f))
        except Exception as e:
            print(f'[skip] {os.path.basename(path)}: {e}')
    return sorted(found)


def main(argv=None):
    share = get_package_share_directory('fbot_recognition')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='*', help='.pt files in the weights folder (default: model_file of config/*.yaml)')
    parser.add_argument('--weights-dir', default=os.path.join(share, 'weights'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--fp32', action='store_true', help='build in FP32 instead of FP16')
    parser.add_argument('--force', action='store_true', help='rebuild engines that already exist')
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print('CUDA is not available on this machine: TensorRT engines cannot be built here.')
        return 1

    models = args.models or models_from_configs(os.path.join(share, 'config'))
    if not models:
        print('No models to export.')
        return 1

    failed = []
    for name in models:
        pt = name if os.path.isabs(name) else os.path.join(args.weights_dir, name)
        engine = os.path.splitext(pt)[0] + '.engine'
        onnx = os.path.splitext(pt)[0] + '.onnx'
        if not os.path.exists(pt):
            print(f'[skip] {pt} not found')
            failed.append(name)
            continue
        if os.path.exists(engine) and not args.force:
            print(f'[skip] {os.path.basename(engine)} already exists (use --force to rebuild)')
            continue

        had_onnx = os.path.exists(onnx)
        print(f'[build] {os.path.basename(pt)} -> {os.path.basename(engine)} '
              f'({"FP32" if args.fp32 else "FP16"}, imgsz {args.imgsz}); this takes a few minutes...')
        t = time.time()
        try:
            YOLO(pt).export(format='engine', half=not args.fp32, imgsz=args.imgsz, device=0, verbose=False)
            print(f'[ok] {engine} ({time.time() - t:.0f} s)')
        except Exception as e:
            print(f'[fail] {os.path.basename(pt)}: {e}')
            failed.append(name)
        finally:
            if not had_onnx and os.path.exists(onnx):
                os.remove(onnx)  # intermediate file of the export

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
