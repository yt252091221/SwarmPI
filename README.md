# SwarmPI - 身份保持多目标追踪

基于零水印身份验证的多目标追踪系统（v6），集成了多种经典追踪算法用于性能对比。

## 项目结构

```
├── V6.py                    # SwarmPI v6 核心追踪器
├── algorithms/              # 对比算法实现
│   ├── bytetrack/           # ByteTrack
│   ├── deepsort/            # DeepSORT
│   ├── ocsort/              # OC-SORT
│   └── boxmot/              # BotSort / StrongSORT (via boxmot)
├── data/
│   ├── full_sequences/      # 完整数据集
│   │   ├── count2/          # （自采集）
│   │   ├── count3/          # （自采集）
│   │   ├── watercraft/      # （自采集）
│   │   ├── uavswarm39/      # （公开数据集，需下载图像）
│   │   ├── uavswarm43/      # （公开数据集，需下载图像）
│   │   └── uavswarm61/      # （公开数据集，需下载图像）
│   └── MOT17-02-FRCNN/      # MOT17 参考序列（公开数据集，需下载图像）
├── requirements.txt
└── README.md
```

## 数据集

数据集采用 MOT Challenge 格式，每个序列目录包含：

- `det.txt` 或 `det/det.txt` - 检测结果
- `gt.txt` 或 `gt/gt.txt` - 标注真值
- `seqinfo.ini` - 序列元信息
- `img1/` - 图像帧

### 自采集数据

`count2`、`count3`、`watercraft` 为自采集数据集。图像文件托管在 [Releases](../../releases) 页面，下载后解压到对应目录的 `img1/` 文件夹：

| 数据集 | 下载文件 | 解压目标 |
|--------|----------|----------|
| count2 | count2_img1.zip | `data/full_sequences/count2/img1/` |
| count3 | count3_img1.zip | `data/full_sequences/count3/img1/` |
| watercraft | watercraft_img1.zip | `data/full_sequences/watercraft/img1/` |

### 公开数据集

以下数据集的标注文件已包含在仓库中，图像需从原始来源下载后放入对应 `img1/` 目录：

| 数据集 | 下载链接 |
|--------|----------|
| MOT17 | [MOT17 Challenge](https://motchallenge.net/data/MOT17/) |
| UAVSwarm | [UAVSwarm Dataset](https://github.com/UAVSwarm/UAVSwarm-dataset) |

下载后将图像帧放入对应目录的 `img1/` 文件夹即可。

## 依赖环境

```bash
pip install -r requirements.txt
```

主要依赖：

- Python 3.8+
- NumPy, OpenCV, SciPy
- PyWavelets (零水印特征)
- filterpy (Kalman 滤波)
- ultralytics (YOLO 检测)
- PyTorch (DeepSORT 特征提取)

## 算法说明

### SwarmPI v6

核心创新点：
- **零水印身份编码**: 基于 DWT 的三值化零水印特征，用于目标身份保持
- **Prototype 独立验证**: 将 Prototype 作为身份验证层，与局部特征（空间/HSV/运动/零水印）互补
- **自适应特征融合**: 在线学习各特征分量的权重

### 对比算法

| 算法 | 说明 |
|------|------|
| ByteTrack | 基于 IoU 关联的高效追踪器 |
| DeepSORT | 结合深度外观特征的追踪器 |
| OC-SORT | 基于观测为中心的追踪器 |
| BotSort | ByteTrack + ReID 外观特征 |
| StrongSORT | 强化版 SORT 追踪器 |

## 引用

如果本项目对您的研究有帮助，请引用相关论文。
