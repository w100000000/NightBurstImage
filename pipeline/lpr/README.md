# HyperLPR3 — License Plate Recognition

## Overview

CPU-based license plate recognition using ncnn inference, running on ELF2 RK3588.

The pipeline is: YOLO detection → CTC recognition → classification → save to TF card.

## Model Files (on target board)

Place these ncnn models on the board at the path passed to `--lpr` (typically `/mnt/sdcard/`):

- `y5fu_320x_sim.ncnn.bin` / `y5fu_320x_sim.ncnn.param` — YOLO license plate detector
- `rpv3_mdict_160_r3.ncnn.bin` / `rpv3_mdict_160_r3.ncnn.param` — CTC recognition model
- `litemodel_cls_96x_r1.ncnn.bin` / `litemodel_cls_96x_r1.ncnn.param` — Classification model

These are exported from HyperLPR3 ONNX models. See the main HyperLPR3 project for conversion scripts.

## Build

Compiled as part of the main pipeline:

```bash
cd ../pipeline/core
make lpr_worker.o
```

The `lpr_worker.c` depends on `pipeline_types.h` from `pipeline/core/`.

## Reference

- HyperLPR3: https://github.com/szad670401/HyperLPR3
