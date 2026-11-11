import os

# Must run before numpy is imported. With the default (one OpenBLAS thread per core),
# the small matrix ops of the YOLO trackers (BoT-SORT/ByteTrack) make every core
# busy-wait: on the Jetson the tracking used ~750% CPU, and ~80% with 1 thread at the
# same FPS. Only OpenBLAS (numpy) is limited; torch/OpenMP threads are untouched.
# Override by exporting OPENBLAS_NUM_THREADS before launching.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from .base_recognition import *
#from .yolov8_recognition import *
# from .face_recognition import *