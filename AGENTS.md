# Repository Guidelines

## Project Structure & Module Organization

This repository deploys BEVFormer, BEVDet, and selected 2D detectors to TensorRT. Core Python conversion and runtime helpers live in `det2trt/`, with model-specific entry points in `tools/bevformer/`, `tools/bevdet/`, and `tools/2d/`. Configuration files are under `configs/`, grouped by model family and shared bases. Shell workflows are in `samples/`, including conversion, quantization, evaluation, and data preparation scripts. Custom TensorRT CUDA/C++ plugins are in `TensorRT/plugin/`, with shared helpers in `TensorRT/common/`. External or vendored code belongs in `third_party/`; datasets and generated symlinks belong in `data/`; checkpoints, ONNX files, and engines belong in `checkpoints/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install project-level Python dependencies after the required CUDA, TensorRT, PyTorch, MMCV, MMDetection, and MMDeploy stack is available.
- `cd TensorRT/build && cmake .. -DCMAKE_TENSORRT_PATH=/path/to/TensorRT && make -j$(nproc) && make install`: build and install custom TensorRT plugins.
- `cd third_party/bev_mmdet3d && python setup.py build develop`: build the BEVDetection3D ops used by BEV workflows.
- `sh samples/test_trt_ops.sh`: run custom TensorRT plugin unit tests.
- `sh samples/bevformer/base/pth2onnx.sh -d 0` then `sh samples/bevformer/base/onnx2trt_fp16.sh -d 0`: example export and TensorRT build flow.

## Coding Style & Naming Conventions

Python is formatted with Black `19.10b0`; run `python3 ci/check/run_py_format.py --source_dir $PWD --fix` before submitting. CI also runs `clang-format -style=file` over `*.cpp`, `*.hpp`, `*.cu`, `*.c`, and `*.h`. Match existing 4-space Python indentation, `snake_case` functions and files, and model/config names such as `bevformer_base_trt_q.py`. Keep sample scripts named by action, for example `pth2onnx.sh`, `onnx2trt_int8.sh`, and `trt_evaluate_fp16.sh`.

## Testing Guidelines

There is no broad pytest suite in this tree. Validate plugin changes with `samples/test_trt_ops.sh`, then run the smallest relevant model workflow: PyTorch evaluation, ONNX export, TensorRT conversion, and TensorRT evaluation from the matching `samples/<model>/` directory. For quantization changes, include INT8 or QDQ sample scripts and compare reported accuracy against the baseline README tables.

## Commit & Pull Request Guidelines

History uses short imperative or descriptive subjects, often prefixed by scope verbs such as `fix`, `support`, `add`, `Rename`, or `Update`, with merged PR numbers appended by GitHub. Keep commits focused, for example `fix BEVDet INT8 calibration path`. PRs should describe the affected model path, list exact commands run, note CUDA/TensorRT versions, and include benchmark or accuracy deltas when inference behavior changes.
