# 天气四分类数据下载与清洗总结

## 1. 官方目标类别

```text
晴 -> sunny
阴 -> cloudy
雨 -> rainy
雪 -> snowy
```

统一训练标签：

```json
{
  "0": "sunny",
  "1": "cloudy",
  "2": "rainy",
  "3": "snowy"
}
```

## 2. 已下载数据集

### WeatherNet-05-18039

来源：

- https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039
- 实际下载使用镜像：https://hf-mirror.com/datasets/prithivMLmods/WeatherNet-05-18039

原始文件：

```text
data/external/raw/weathernet05_0000.parquet
data/external/raw/weathernet05_0001.parquet
data/external/raw/weathernet05_README.md
```

原始标签：

| Source Label | Source Name | Count | Action |
|---:|---|---:|---|
| 0 | cloudy or overcast | 6702 | map to cloudy |
| 1 | foggy or hazy | 1261 | drop |
| 2 | rain or storm | 1927 | map to rainy |
| 3 | snow or frosty | 1875 | map to snowy |
| 4 | sun or clear | 6274 | map to sunny |

## 3. 暂不可自动下载的数据集

### Roboflow Weather Classification Dataset

状态：

```text
暂未下载
```

原因：

- Roboflow Universe 页面被 Cloudflare challenge 拦截。
- Roboflow API 明确要求 API key。

后续处理：

```text
如果提供 Roboflow API key，可继续下载并合并。
```

### WEAPD / Weather Image Recognition

状态：

```text
暂未下载
```

原因：

- Harvard Dataverse 原始文件可访问。
- 文件为 dataset(WEAPD).rar，约 621MB。
- 当前环境没有 7z、unrar、WinRAR 或 Python RAR 解压依赖。

后续处理：

```text
安装或提供 RAR 解压工具后，可下载并只提取 rain / snow 类作为补充。
```

### Kaggle 数据集

状态：

```text
暂未下载
```

原因：

- 当前未发现 kaggle.json。
- Kaggle API / 直链下载不可用。

后续处理：

```text
如果配置 Kaggle API token，可继续下载 Kaggle 数据集。
```

## 4. 清洗规则

执行脚本：

```text
scripts/prepare_weathernet4.py
```

清洗动作：

```text
1. 从 parquet 解码图片
2. 删除 foggy / hazy 类
3. 映射到 sunny / cloudy / rainy / snowy 四类
4. 校验图片可打开
5. 统一导出为 RGB JPEG
6. 删除精确重复图片
7. 记录同图不同标签冲突
8. 生成严格版 CSV，过滤冲突 md5
9. 分层划分 train / val
```

## 5. 清洗结果

普通清洗结果：

```text
原始图片数: 18039
保留四类图片: 16562
删除 foggy/hazy: 1261
删除精确重复: 216
同图不同标签冲突: 29
```

严格版索引结果：

```text
严格版保留图片: 16533
训练集: 13226
验证集: 3307
验证比例: 20%
划分方式: stratified split
随机种子: 42
```

严格版类别分布：

| Class | Total | Train | Val |
|---|---:|---:|---:|
| cloudy | 6486 | 5189 | 1297 |
| rainy | 1925 | 1540 | 385 |
| snowy | 1875 | 1500 | 375 |
| sunny | 6247 | 4997 | 1250 |

## 6. 推荐训练入口

后续训练优先使用严格版 CSV：

```text
data/processed/weather4/train_strict.csv
data/processed/weather4/val_strict.csv
data/processed/weather4/metadata_strict.csv
data/processed/weather4/label_map.json
```

图片目录：

```text
data/processed/weather4/images/cloudy
data/processed/weather4/images/rainy
data/processed/weather4/images/snowy
data/processed/weather4/images/sunny
```

视觉抽样图：

```text
data/processed/weather4/reports/samples/cloudy_samples.jpg
data/processed/weather4/reports/samples/rainy_samples.jpg
data/processed/weather4/reports/samples/snowy_samples.jpg
data/processed/weather4/reports/samples/sunny_samples.jpg
```

## 7. 质量判断

当前可用于第一版 baseline 训练。

注意事项：

- `cloudy` 类存在少量晴天/日落边界样本，后续建议做人工抽样二次筛。
- `rainy` 类质量较好，雨、伞、积水、雨刷等特征明显。
- `snowy` 类质量较好，包含雪景、道路积雪、霜冻等样本。
- `sunny` 类整体可用，主要是晴朗天空和清晰光照场景。

## 8. 下一步建议

```text
1. 使用 train_strict.csv / val_strict.csv 跑通 PyTorch baseline
2. 先训练 ResNet50 或 EfficientNet-B0
3. 验证指标使用 macro-F1
4. 输出混淆矩阵，重点观察 cloudy 与 sunny 的混淆
5. 官方训练集发布后，用官方训练集重新划分验证集并微调
```

## 9. 本地目录型数据清洗结果

用户提供的本地目录：

```text
data/data
```

目录结构：

```text
data/data/cloudy
data/data/foggy
data/data/rainy
data/data/snowy
data/data/sunny
```

该目录与 WeatherNet 原始规模一致，共 18,039 张图片。已执行非破坏式清洗：不删除原图，只生成严格版训练索引。

清洗脚本：

```text
scripts/prepare_weather4_from_dir.py
```

清洗输出：

```text
data/processed/weather4_from_dir
```

清洗规则：

```text
cloudy -> 阴
rainy -> 雨
snowy -> 雪
sunny -> 晴
foggy -> 删除，不进入训练
```

严格版结果：

```text
原始图片数: 18039
删除 foggy: 1261
删除同标签精确重复: 186
过滤同图不同标签冲突图片: 59
严格版可用图片: 16533
训练集: 13226
验证集: 3307
```

严格版类别分布：

| Class | Total | Train | Val |
|---|---:|---:|---:|
| cloudy | 6486 | 5189 | 1297 |
| rainy | 1925 | 1540 | 385 |
| snowy | 1875 | 1500 | 375 |
| sunny | 6247 | 4997 | 1250 |

当前推荐应用入口：

```text
data/processed/weather4_from_dir/train_strict.csv
data/processed/weather4_from_dir/val_strict.csv
data/processed/weather4_from_dir/metadata_strict.csv
data/processed/weather4_from_dir/label_map.json
```

说明：

- `weather4_from_dir` 的 CSV 直接引用 `data/data/...` 中的原图路径，不重复复制图片。
- 训练脚本读取 CSV 时，应以项目根目录 `C:/Users/Z Pengfei/Desktop/caip` 作为路径根。
- 如果后续训练代码只支持图片目录，也可以使用上一轮已导出图片的 `data/processed/weather4/images`。
