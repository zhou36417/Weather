# 天气四分类训练代码说明

## 1. 已实现文件

```text
scripts/weather4_lib.py
scripts/train_weather4.py
scripts/predict_weather4.py
scripts/split_weather4_dataset.py
scripts/evaluate_weather4.py
```

代码目标：

- 读取清洗后的四分类 CSV。
- 使用 PyTorch / TorchVision 构建分类模型。
- 按验证集 `macro-F1` 保存最佳模型。
- 输出每轮训练日志、指标和混淆矩阵。
- 支持从 metadata 生成 train / val / test 三划分。
- 支持在本地 test.csv 上做最终评估。
- 支持加载 checkpoint 对图片或目录做预测。

## 2. 当前训练数据入口

推荐使用目录型清洗结果：

```text
data/processed/weather4_from_dir/train_strict.csv
data/processed/weather4_from_dir/val_strict.csv
data/processed/weather4_from_dir/label_map.json
```

当前已新增三划分数据入口，更推荐后续训练使用：

```text
data/processed/weather4_from_dir_70_15_15/train.csv
data/processed/weather4_from_dir_70_15_15/val.csv
data/processed/weather4_from_dir_70_15_15/test.csv
data/processed/weather4_from_dir_70_15_15/trainval.csv
data/processed/weather4_from_dir_70_15_15/all.csv
data/processed/weather4_from_dir_70_15_15/label_map.json
```

三划分比例：

```text
train: 70%
val: 15%
test: 15%
```

三划分数量：

| Class | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| cloudy | 6486 | 4540 | 973 | 973 |
| rainy | 1925 | 1347 | 289 | 289 |
| snowy | 1875 | 1313 | 281 | 281 |
| sunny | 6247 | 4373 | 937 | 937 |

补充索引：

```text
trainval.csv: train + val，共 14,053 张
all.csv: 当前严格清洗后的全部可用数据，共 16,533 张
```

使用原则：

```text
调参阶段：train.csv 训练，val.csv 选模型，test.csv 不动
定参后：trainval.csv 训练，再用 test.csv 做一次最终本地评估
最终提交前：如果不再需要本地无偏测试，可用 all.csv 训练最终权重
```

注意：

```text
不要同时使用 data/data 和 data/processed/weather4/images。
它们来自同一份 WeatherNet 数据，叠加会造成重复训练和数据泄漏。
```

类别：

```text
0: sunny
1: cloudy
2: rainy
3: snowy
```

## 3. 环境说明

当前可用环境：

```text
D:/Anaconda/python.exe
torch 2.11.0+cpu
torchvision 0.26.0+cpu
CUDA: unavailable
timm: unavailable
```

因此当前代码先基于 TorchVision 模型落地：

```text
resnet50
efficientnet_b0
efficientnet_b3
convnext_tiny
```

本机 Anaconda 存在 OpenMP 重复运行时问题，训练和预测脚本已在进程内设置兼容开关。

当前机器检测结果：

```text
torch 2.11.0+cpu
cuda_available: False
cuda_device_count: 0
```

如果切换到 CUDA 版 PyTorch 环境，代码可直接使用 GPU。

## 4. Smoke Test

已完成最小链路测试：

```text
模型: resnet50
权重: none
epoch: 1
batch size: 2
image size: 64
train batches: 1
val batches: 1
device: cpu
```

输出目录：

```text
runs/weather4_smoke
```

已生成：

```text
best_model.pth
last_model.pth
best_metrics.json
best_confusion.csv
training_log.csv
predict_smoke.csv
```

注意：smoke test 使用随机初始化和极少 batch，只验证代码链路，不代表模型效果。

## 5. 正式 Baseline 训练命令

CPU 环境下建议先跑小模型：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/train_weather4.py" `
  --project-root "C:/Users/Z Pengfei/Desktop/caip" `
  --train-csv "data/processed/weather4_from_dir_70_15_15/train.csv" `
  --val-csv "data/processed/weather4_from_dir_70_15_15/val.csv" `
  --label-map "data/processed/weather4_from_dir_70_15_15/label_map.json" `
  --output-dir "runs/weather4_resnet50_baseline" `
  --model resnet50 `
  --weights imagenet `
  --epochs 10 `
  --batch-size 16 `
  --image-size 224 `
  --num-workers 0 `
  --threads 4 `
  --class-weights balanced `
  --device cpu
```

如果 ImageNet 权重无法下载，可先用随机权重验证流程：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/train_weather4.py" `
  --project-root "C:/Users/Z Pengfei/Desktop/caip" `
  --train-csv "data/processed/weather4_from_dir_70_15_15/train.csv" `
  --val-csv "data/processed/weather4_from_dir_70_15_15/val.csv" `
  --label-map "data/processed/weather4_from_dir_70_15_15/label_map.json" `
  --output-dir "runs/weather4_resnet50_random" `
  --model resnet50 `
  --weights none `
  --epochs 3 `
  --batch-size 16 `
  --image-size 224 `
  --num-workers 0 `
  --threads 4 `
  --class-weights balanced `
  --device cpu
```

有 GPU 后推荐：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/train_weather4.py" `
  --project-root "C:/Users/Z Pengfei/Desktop/caip" `
  --train-csv "data/processed/weather4_from_dir_70_15_15/train.csv" `
  --val-csv "data/processed/weather4_from_dir_70_15_15/val.csv" `
  --label-map "data/processed/weather4_from_dir_70_15_15/label_map.json" `
  --output-dir "runs/weather4_convnext_tiny" `
  --model convnext_tiny `
  --weights imagenet `
  --epochs 15 `
  --batch-size 32 `
  --image-size 224 `
  --num-workers 4 `
  --class-weights balanced `
  --device auto `
  --amp
```

GPU 版训练的前提：

```text
1. 当前 Python 环境必须安装 CUDA 版 torch / torchvision
2. torch.cuda.is_available() 必须为 True
3. 当前机器或远程环境必须有 NVIDIA GPU 和匹配驱动
```

确认 GPU 可用：

```powershell
& "D:/Anaconda/python.exe" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 6. 预测命令

单图预测：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/predict_weather4.py" `
  --checkpoint "C:/Users/Z Pengfei/Desktop/caip/runs/weather4_resnet50_baseline/best_model.pth" `
  --input "C:/Users/Z Pengfei/Desktop/caip/data/data/cloudy/10161882.jpg" `
  --output-csv "C:/Users/Z Pengfei/Desktop/caip/runs/weather4_resnet50_baseline/predict_one.csv" `
  --device cpu
```

目录预测：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/predict_weather4.py" `
  --checkpoint "C:/Users/Z Pengfei/Desktop/caip/runs/weather4_resnet50_baseline/best_model.pth" `
  --input "C:/Users/Z Pengfei/Desktop/caip/data/data" `
  --output-csv "C:/Users/Z Pengfei/Desktop/caip/runs/weather4_resnet50_baseline/predictions.csv" `
  --batch-size 32 `
  --device cpu
```

## 7. 输出文件说明

每次训练输出目录会包含：

```text
config.json
training_log.csv
epoch_XXX_metrics.json
epoch_XXX_confusion.csv
best_model.pth
best_metrics.json
best_confusion.csv
last_model.pth
```

核心评估指标：

```text
best_metrics.json -> macro_f1
best_confusion.csv -> 混淆矩阵
training_log.csv -> 每轮 val_macro_f1
```

## 8. 测试集评估

训练完成后，使用 test.csv 做最终本地评估：

```powershell
& "D:/Anaconda/python.exe" "C:/Users/Z Pengfei/Desktop/caip/scripts/evaluate_weather4.py" `
  --checkpoint "C:/Users/Z Pengfei/Desktop/caip/runs/weather4_convnext_tiny/best_model.pth" `
  --project-root "C:/Users/Z Pengfei/Desktop/caip" `
  --csv "data/processed/weather4_from_dir_70_15_15/test.csv" `
  --label-map "data/processed/weather4_from_dir_70_15_15/label_map.json" `
  --output-dir "runs/weather4_convnext_tiny/test_eval" `
  --batch-size 32 `
  --num-workers 4 `
  --device auto
```

测试集只用于最终本地评估，不要用它反复调参。

## 9. 下一步优化方向

```text
1. 先跑 resnet50 / efficientnet_b0 baseline
2. 观察 best_confusion.csv
3. 重点看 cloudy vs sunny、cloudy vs rainy
4. 如果本地 macro-F1 稳定，再尝试 efficientnet_b3 / convnext_tiny
5. 官方训练集发布后，替换 CSV 并继续微调
```
