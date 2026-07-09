# models · 部署模型

板端运行所需的模型文件（拷贝到板子 `/mnt/sdcard/` 即可运行）。

## 目录结构

```
models/
├── denoise/    暗光去噪增强（NPU）
│   └── nbinet_272x480_fp16.rknn
├── lpr/        车牌检测+识别（CPU/ncnn）
│   ├── y5fu_320x_sim.ncnn.{param,bin}       # 车牌检测（YOLO）
│   ├── rpv3_mdict_160_r3.ncnn.{param,bin}   # 车牌字符识别（CTC）
│   └── litemodel_cls_96x_r1.ncnn.{param,bin}# 单/双层分类
└── README.md
```

| 用途 | 框架/精度 | 来源 |
|------|-----------|------|
| 暗光去噪增强 | RKNN / FP16（RK3588 NPU） | 自研 NBINet（基于 [D2HNet](https://github.com/zhaoyuzhi/D2HNet) 改造） |
| 车牌检测（YOLO） | ncnn / CPU | [HyperLPR3](https://github.com/szad670401/HyperLPR)（Apache-2.0） |
| 车牌字符识别（CTC） | ncnn / CPU | HyperLPR3（Apache-2.0） |
| 车牌单/双层分类 | ncnn / CPU | HyperLPR3（Apache-2.0） |

## 部署

```bash
scp models/denoise/* models/lpr/* root@<BOARD_IP>:/mnt/sdcard/
```

## 许可与致谢

- 车牌检测/识别/分类模型来自 **HyperLPR3**（Apache-2.0），版权归原作者所有，本仓库依据其许可再分发并致谢。
- 去噪模型 NBINet 由本团队自建低照度数据集训练，网络结构参考 D2HNet。
