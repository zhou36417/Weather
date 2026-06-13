# 天气分类竞赛高分路线图与公开训练集调研

## 1. 当前约束

官方当前说明：

- 任务是天气图像分类。
- 最终成绩为 `F1_score * 100`。
- 官方后续只提供训练集。
- 官方不提供测试集标签。
- 官方不单独提供验证集。
- 当前阶段官方训练集尚未发布。

因此当前工作的重点不是直接训练最终模型，而是先完成三件事：

1. 找高可用、高质量、类别接近的公开天气图像数据集。
2. 先搭建可复用的训练、验证、推理、F1 评估流程。
3. 等官方训练集发布后，快速迁移到官方数据上做最终微调。

重要假设：

- 如果比赛规则允许使用外部数据，可以将公开数据集作为补充训练数据或预训练数据。
- 如果比赛规则禁止外部数据，公开数据集只能用于前期方案验证、模型结构选择和代码流程调试，最终模型必须只用官方训练集训练。
- 在规则未明确前，默认按保守策略处理：公开数据集优先用于预训练和流程验证，最终以官方训练集为准。

## 2. 高分核心原则

本竞赛的核心指标是 F1，不是单纯 accuracy。

高分路线应围绕以下闭环：

```text
公开数据预训练/验证流程
-> 官方训练集发布
-> 从官方训练集中划分验证集
-> 按 macro-F1 保存最优模型
-> 混淆矩阵定位低分类别
-> 针对性调数据增强、类别权重、模型结构
-> TTA / KFold / 集成冲分
-> 控制推理时间并提交
```

重点：

- 必须自己从官方训练集中划分验证集。
- 必须使用分层划分，保证每个天气类别在训练集和验证集中的比例接近。
- 必须按验证集 `macro-F1` 保存模型，而不是按 loss 或 accuracy。
- 必须关注每个类别的 F1，尤其是少数类和易混类别。

## 3. 公开训练集优先级

### 3.1 第一优先级：WeatherNet-05-18039

链接：

- https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039

关键信息：

- Hugging Face 数据集。
- 图像分类任务。
- 18,039 张图像。
- 5 个天气类别。
- Apache-2.0 许可证。
- 文件约 544 MB。
- 可用 `datasets` 库直接加载。

推荐用途：

- 当前阶段优先下载和使用。
- 用于搭建 PyTorch 训练流程。
- 用于验证 `train.py`、`valid.py`、`predict.py`。
- 用于训练第一版天气分类预训练权重。

优点：

- 可访问性较好。
- 数据量适中，适合快速迭代。
- 类别数量接近一般天气分类任务。
- 许可证较友好。

风险：

- 具体类别需要下载后完整核验。
- Hugging Face 页面预览显示部分标签，但不一定完整展示所有类别。
- 与官方数据分布可能不同，不能直接代表线上效果。

优先级结论：

```text
最推荐作为当前第一训练集。
```

### 3.2 第二优先级：Weather Image Recognition / WEAPD

链接：

- Kaggle 镜像：https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
- 原始引用：Weather phenomenon database, DOI: https://doi.org/10.7910/DVN/M8JQCR

关键信息：

- 约 6,862 张天气图像。
- 11 类：dew、fog/smog、frost、glaze、hail、lightning、rain、rainbow、rime、sandstorm、snow。
- Kaggle 页面标注为 CC0 Public Domain。
- 偏天气现象识别，尤其是雨、雪、雾、霜、冰雹、沙尘等复杂天气。

推荐用途：

- 用于增强恶劣天气类别的识别能力。
- 用于训练模型对雾、雪、雨、沙尘、冰雹等细粒度天气特征的感知。
- 用于补充 WeatherNet 中可能不足的极端天气样本。

优点：

- 类别细，天气现象明显。
- 适合提升复杂天气条件下的鲁棒性。
- 公开引用较多，适合作为研究型数据源。

风险：

- 类别与比赛官方类别未必完全一致。
- 如果官方只分晴、雨、雪、雾、阴等大类，需要做类别映射。
- 部分图像可能来自互联网，仍需抽样检查标签质量。

建议类别映射：

```text
rain -> rain
snow -> snow
fog/smog -> fog
sandstorm -> dust/sand
hail/frost/rime/glaze -> snow/ice/adverse_weather
rainbow/lightning/dew -> 视官方类别决定是否使用
```

优先级结论：

```text
适合作为恶劣天气补充集，不建议单独作为唯一训练集。
```

### 3.3 第三优先级：Mendeley Multi-class Weather Dataset

链接：

- https://data.mendeley.com/datasets/4drtyfjtfy

关键信息：

- 数据集名称：Multi-class Weather Dataset for Image Classification。
- DOI：10.17632/4drtyfjtfy.1。
- 许可证：CC BY 4.0。
- 文件 `dataset2.zip` 约 91.2 MB。
- 用于多类别天气图像识别研究。

推荐用途：

- 作为轻量补充数据集。
- 用于快速做模型 sanity check。
- 用于扩充早期 baseline。

优点：

- 学术数据源，引用清晰。
- 文件较小，下载和实验成本低。
- 适合快速验证流程。

风险：

- 数据量可能有限。
- 类别、图片质量和目录结构需要下载后核验。
- CC BY 4.0 要求保留署名和来源说明。

优先级结论：

```text
适合作为补充训练集和快速验证集，不作为主数据源。
```

### 3.4 第四优先级：BDD100K / RWVC-BDD100K

链接：

- BDD100K 说明：https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html
- RWVC-BDD100K 注释：https://github.com/enricivi/RWVC-BDD100K

关键信息：

- BDD100K 是自动驾驶场景大规模数据集。
- 包含 100K 图像，覆盖城市、街道、住宅区、高速路等场景。
- 图像包含不同天气、不同时间段。
- RWVC-BDD100K 是基于 BDD100K 的道路、天气、能见度图像级注释子集。
- RWVC-BDD100K 子集约 13.3K 图像，划分为 8.8K 训练、2.5K 验证、2K 测试。

推荐用途：

- 如果官方任务偏交通气象、自动驾驶、道路场景，优先考虑。
- 用于训练模型识别道路场景中的晴、雨、雪、雾、阴天等。
- 用作领域预训练数据。

优点：

- 图像质量和标注规范较强。
- 与赛题描述中的自动驾驶场景高度相关。
- 适合提升模型在真实道路天气中的泛化能力。

风险：

- BDD100K 需要手动注册和下载。
- 数据体积较大。
- 许可证和使用范围需要单独确认。
- RWVC-BDD100K 主要是注释，仍依赖 BDD100K 原始图像。

优先级结论：

```text
适合做领域增强，不适合当前第一时间快速起步。
```

### 3.5 第五优先级：Kaggle Multi-class Weather Dataset

链接：

- https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset

关键信息：

- 常见四类天气图像集。
- 常见类别为 Cloudy、Rain、Shine、Sunrise。
- 数据量较小。

推荐用途：

- 快速跑通分类代码。
- 做最小 baseline。
- 用于检查数据加载、训练、保存、推理流程。

优点：

- 简单。
- 易理解。
- 适合新建项目时快速验证代码正确性。

风险：

- 类别少。
- Sunrise 不一定是官方天气类别。
- 数据量较小，泛化能力有限。
- 不能作为高分主训练集。

优先级结论：

```text
只建议用于代码调试，不建议作为高分主数据源。
```

## 4. 当前阶段推荐数据组合

在官方训练集未发布前，推荐按如下顺序准备：

```text
主数据集：
WeatherNet-05-18039

恶劣天气补充：
Weather Image Recognition / WEAPD

轻量辅助：
Mendeley Multi-class Weather Dataset

场景增强：
BDD100K / RWVC-BDD100K，视下载成本和官方场景而定
```

推荐先做两个版本：

### 版本 A：快速可用版

```text
WeatherNet-05-18039
-> stratified train/val split
-> ConvNeXt-Tiny / EfficientNet-B3
-> macro-F1 评估
```

用途：

- 快速搭建完整训练流程。
- 先得到一个稳定可复现的基线。

### 版本 B：高质量补充版

```text
WeatherNet-05-18039
+ WEAPD
+ Mendeley MWD
-> 类别映射
-> 数据清洗
-> 预训练
-> 等官方训练集发布后微调
```

用途：

- 提升模型对复杂天气的泛化能力。
- 为官方训练集发布后的快速微调做准备。

## 5. 数据质量筛选标准

公开数据下载后必须先做清洗，不能直接混合训练。

必须检查：

- 图片是否损坏。
- 图片是否重复。
- 类别是否严重不均衡。
- 标签是否明显错误。
- 图像是否包含水印、拼图、图标、非真实照片。
- 类别定义是否和官方一致。
- 是否存在训练集与未来官方训练集的重复图。

建议建立数据清洗脚本：

```text
1. 扫描所有图片是否可打开
2. 统计每类数量
3. 生成每类样本预览图
4. 计算感知哈希，去除重复图
5. 抽样人工检查标签
6. 输出 clean_dataset.csv
```

## 6. 官方训练集发布后的处理流程

官方训练集发布后，不要立刻全量训练提交。

应按以下步骤：

```text
1. 读取官方训练集目录和标签
2. 统计类别数量
3. 分层划分训练集和验证集
4. 先用 ImageNet 预训练模型训练 baseline
5. 再加载公开数据预训练权重做微调
6. 对比两者验证 macro-F1
7. 只保留能提升官方验证集 F1 的方案
8. 最终用 KFold 或全量官方训练集训练提交模型
```

验证集建议：

```text
数据较少：5-fold StratifiedKFold
数据中等：train/val = 8/2
类别极不均衡：StratifiedKFold + class weight
```

保存模型标准：

```text
best_model.pth = 验证集 macro-F1 最高的模型
```

不要按以下指标单独保存：

```text
最低 val_loss
最高 accuracy
最后一轮 epoch
```

## 7. 模型路线

推荐框架：

```text
PyTorch + torchvision + timm
```

推荐模型顺序：

```text
baseline:
ResNet50 / EfficientNet-B0

主力:
ConvNeXt-Tiny / EfficientNet-B3

冲分:
ConvNeXt-Small / Swin-Tiny
```

推荐训练设置：

```text
loss: CrossEntropyLoss
optimizer: AdamW
scheduler: warmup + cosine
metric: macro-F1
image_size: 224 -> 320 -> 384
batch_size: 根据显存调整
```

类别不均衡时再尝试：

```text
class weight
WeightedRandomSampler
Focal Loss
```

原则：

```text
每次只改一个变量。
只相信官方验证集或从官方训练集中划分出的验证集。
公开数据提升不了官方验证 F1，就不要合入最终训练。
```

## 8. 数据增强策略

推荐增强：

```text
RandomResizedCrop
HorizontalFlip
ColorJitter，幅度适中
RandomRotation，小角度
RandAugment / TrivialAugment
Normalize
```

谨慎增强：

```text
强模糊
强颜色扰动
VerticalFlip
过强 Mixup / CutMix
```

原因：

天气分类依赖天空颜色、能见度、云层纹理、雨雪雾痕迹、地面积水和光照状态。增强太强会破坏真实天气特征。

## 9. 冲分策略

推荐优先级：

```text
1. 可靠验证集
2. 高质量公开数据预训练
3. 官方训练集微调
4. 每类 F1 + 混淆矩阵错误分析
5. 提高输入分辨率
6. TTA
7. KFold
8. 多模型集成
```

TTA 推荐：

```text
原图
水平翻转
多尺度 resize，谨慎使用
```

集成推荐：

```text
ConvNeXt-Tiny + EfficientNet-B3
3-fold models average logits
```

注意：

如果文档中同分会比较推理时间，那么最终不要无限堆模型。建议准备两个提交版本：

```text
高分版：KFold + TTA + 集成
稳定版：单模型 ConvNeXt-Tiny / EfficientNet-B3 + TTA
```

## 10. 当前立即执行路线

在官方训练集未发布前，建议立即做：

```text
1. 下载 WeatherNet-05-18039
2. 下载 WEAPD / Weather Image Recognition
3. 可选下载 Mendeley MWD
4. 建立统一数据目录
5. 写数据扫描和清洗脚本
6. 建立 stratified split
7. 跑通 PyTorch baseline
8. 输出 macro-F1、每类 F1、混淆矩阵
9. 保存公开数据预训练权重
10. 等官方训练集发布后迁移微调
```

统一目录建议：

```text
data/
  external/
    weathernet05/
    weapd/
    mwd/
  official/
    train/
  processed/
    train.csv
    val.csv
    label_map.json
  checkpoints/
```

## 11. 最终结论

当前阶段最高收益的事情不是直接追求复杂模型，而是：

```text
先找到高质量公开天气数据
-> 搭建可复用训练和 F1 验证流程
-> 用公开数据训练一个强初始化模型
-> 官方训练集发布后快速做分层验证和微调
-> 用 macro-F1、混淆矩阵和 TTA/KFold 冲分
```

数据集优先级结论：

```text
1. WeatherNet-05-18039：当前首选
2. WEAPD / Weather Image Recognition：恶劣天气补充
3. Mendeley MWD：轻量补充
4. BDD100K / RWVC-BDD100K：交通天气领域增强
5. Kaggle Multi-class Weather Dataset：代码调试用
```

最关键的一点：

```text
外部数据只能提高初始能力，最终能否高分必须以官方训练集划分出的验证集 macro-F1 为准。
```

## 12. 参考链接

- WeatherNet-05-18039: https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039
- Weather Image Recognition: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
- WEAPD DOI: https://doi.org/10.7910/DVN/M8JQCR
- Mendeley Multi-class Weather Dataset: https://data.mendeley.com/datasets/4drtyfjtfy
- BDD100K FiftyOne 说明: https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html
- RWVC-BDD100K: https://github.com/enricivi/RWVC-BDD100K
- Kaggle Multi-class Weather Dataset: https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset
- PyTorch Transfer Learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- TorchVision Models: https://pytorch.org/vision/stable/models.html
- scikit-learn f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- timm: https://huggingface.co/docs/timm/index

## 13. 官方四分类数据集筛选结果

官方当前目标类别：

```text
晴
阴
雨
雪
```

英文标签统一映射建议：

```text
sunny / sun / shine / clear -> 晴
cloudy / overcast / partly cloudy -> 阴
rain / rainy / storm -> 雨
snow / snowy -> 雪
fog / foggy / hazy / fogsmog -> 删除，不并入阴
sunrise -> 原则上删除，不直接并入晴
frost / rime / glaze / hail -> 暂不直接并入雪，除非官方训练集也包含类似冰雪场景
```

关键原则：

- 宁可少用，也不要把语义不准的类别强行映射进去。
- `foggy/hazy` 和 `cloudy/overcast` 视觉特征不同，不建议把雾归为阴。
- `sunrise` 更像特殊时间段和光照场景，不等同于普通晴天。
- 外部数据最终是否合入训练，必须看官方训练集划分出的验证集 `macro-F1` 是否提升。

### 13.1 直接可用数据集

#### A. WeatherNet-05-18039

链接：

- https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039

可用性判断：

```text
推荐级别：最高
使用方式：直接使用四个目标类，删除 foggy / hazy 类
适合作为：当前主训练集、预训练集、流程验证集
```

类别映射：

```text
sun / clear -> 晴
cloudy / overcast -> 阴
rain / storm -> 雨
snow / frosty -> 雪
foggy / hazy -> 删除
```

优点：

- 数据量较大，约 18,039 张图像。
- 类别与官方四分类高度接近。
- 许可证较友好。
- Hugging Face 数据集加载方便。

风险：

- 标签命名和实际目录结构需要下载后核验。
- `snow / frosty` 中的 frosty 是否适合并入雪，需要抽样检查。
- 与官方训练集分布可能不同，不能直接代表最终线上效果。

当前结论：

```text
第一优先下载和清洗。
```

#### B. Roboflow Weather Classification Dataset

链接：

- https://universe.roboflow.com/university-of-wolverhampton/weather-classification-2

可用性判断：

```text
推荐级别：高
使用方式：使用 sunny / cloudy / rainy / snowy，删除 foggy
适合作为：第二主数据集、增强数据集、对照验证集
```

类别映射：

```text
sunny -> 晴
cloudy -> 阴
rainy -> 雨
snowy -> 雪
foggy -> 删除
```

优点：

- 类别与官方四分类几乎完全匹配。
- 数据集页面显示约 9.3K 图像。
- Roboflow 支持导出多种格式，便于整理成分类目录或 CSV。

风险：

- 下载通常需要 Roboflow 账号或 API。
- 许可证、版本和导出格式需要下载前确认。
- 需要抽样检查图片是否有水印、重复图和错标图。

当前结论：

```text
第二优先下载。若下载流程顺利，可与 WeatherNet 一起作为当前主训练数据。
```

#### C. FWID Weather Dataset

可用性判断：

```text
推荐级别：中高
使用方式：使用 cloudy / rainy / snowy / sunny，删除 foggy
适合作为：类别均衡补充集
```

类别映射：

```text
sunny -> 晴
cloudy -> 阴
rainy -> 雨
snowy -> 雪
foggy -> 删除
```

优点：

- 常见描述为 5 类天气数据，类别贴近官方任务。
- 如果每类样本均衡，对提升 macro-F1 有帮助。

风险：

- 常见来源需要 IEEE DataPort 或第三方镜像，下载门槛可能较高。
- 许可证和再使用规则必须确认。
- 可访问性不如 Hugging Face 或 Kaggle 稳定。

当前结论：

```text
作为候选补充集。能稳定下载并确认许可证后再纳入。
```

### 13.2 部分可用数据集

#### D. Kaggle Multi-class Weather Dataset

链接：

- https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset

可用性判断：

```text
推荐级别：中
使用方式：只使用 cloudy / rain / shine，删除 sunrise
适合作为：代码调试、晴阴雨补充
不适合作为：完整四分类主训练集
```

类别映射：

```text
shine -> 晴
cloudy -> 阴
rain -> 雨
sunrise -> 删除
snow -> 无
```

优点：

- 下载和理解成本低。
- 适合快速跑通分类训练代码。

风险：

- 缺少雪类。
- `sunrise` 不建议作为晴类使用。
- 数据量偏小，不能承担主训练任务。

当前结论：

```text
只作为流程调试和部分类别补充，不进入第一批主训练集。
```

#### E. Mendeley Multi-class Weather Dataset

链接：

- https://data.mendeley.com/datasets/4drtyfjtfy

可用性判断：

```text
推荐级别：中
使用方式：根据下载后的目录，只保留可映射到晴、阴、雨的类别
适合作为：轻量补充集
不适合作为：完整四分类主训练集
```

可能映射：

```text
shine -> 晴
cloudy -> 阴
rain -> 雨
sunrise -> 删除
snow -> 通常无
```

优点：

- 学术数据源，引用清晰。
- 文件较小，适合快速验证。

风险：

- 大概率缺少雪类。
- CC BY 4.0 需要保留来源署名。
- 类别结构需要下载后核验。

当前结论：

```text
作为备选补充，不作为主数据集。
```

#### F. Weather Image Recognition / WEAPD

链接：

- https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
- https://doi.org/10.7910/DVN/M8JQCR

可用性判断：

```text
推荐级别：中
使用方式：主要保留 rain / snow，其他类别谨慎处理
适合作为：雨雪增强集、恶劣天气补充集
不适合作为：完整四分类主训练集
```

类别映射：

```text
rain -> 雨
snow -> 雪
fog/smog -> 删除
sandstorm -> 删除
dew / frost / glaze / hail / lightning / rainbow / rime -> 默认删除
```

可选实验映射：

```text
frost / rime / glaze -> 雪或冰雪类
```

但只有在官方训练集中的雪类也包含霜、冰、雾凇等场景时，才考虑使用。

优点：

- 雨雪等天气现象明显。
- 可补充复杂天气图像。

风险：

- 没有晴和阴。
- 部分类别不是官方目标类别。
- 强行映射会引入标签噪声。

当前结论：

```text
只作为雨雪补充集。不要作为主训练集。
```

### 13.3 领域补充数据集

#### G. BDD100K / RWVC-BDD100K

链接：

- https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html
- https://github.com/enricivi/RWVC-BDD100K

可用性判断：

```text
推荐级别：中
使用方式：只在官方数据偏交通、道路、自动驾驶场景时使用
适合作为：道路天气领域增强集
不适合作为：当前第一主数据集
```

类别映射：

```text
clear -> 晴
overcast / partly cloudy -> 阴
rainy -> 雨
snowy -> 雪
foggy / undefined -> 删除
```

优点：

- 与赛题描述中的自动驾驶、交通气象场景相关。
- 图像质量和标注体系相对规范。

风险：

- 下载体积大。
- 依赖 BDD100K 原始图像和对应标注。
- 领域偏道路，可能与官方通用天气图像分布不一致。
- 许可证和使用规则需要单独确认。

当前结论：

```text
暂不作为第一批数据。等官方训练集发布后，如果官方图片明显偏道路/交通场景，再考虑引入。
```

### 13.4 暂不推荐数据集

以下数据集当前不作为主力：

```text
DAWN / ACDC 等恶劣天气或自动驾驶数据集
```

原因：

- 多数不是标准图像分类任务。
- 常用于检测、分割或自动驾驶鲁棒性研究。
- 类别不完整，通常没有晴/阴/雨/雪四类完整标签。
- 下载和整理成本高。

### 13.5 当前第一批推荐下载清单

第一批只选高可用、高匹配数据：

```text
1. WeatherNet-05-18039
2. Roboflow Weather Classification Dataset
```

第二批作为补充：

```text
3. FWID Weather Dataset，前提是下载和许可证确认无问题
4. Weather Image Recognition / WEAPD，只取 rain / snow
5. Kaggle Multi-class Weather Dataset，只作代码调试和晴阴雨补充
```

暂缓：

```text
6. BDD100K / RWVC-BDD100K，等官方训练集发布后再决定
```

### 13.6 当前建议的数据合并策略

先建立四类统一标签：

```text
0: sunny
1: cloudy
2: rainy
3: snowy
```

中文展示：

```text
sunny -> 晴
cloudy -> 阴
rainy -> 雨
snowy -> 雪
```

第一阶段训练数据：

```text
WeatherNet-05-18039 四类
+ Roboflow Weather Classification Dataset 四类
```

第二阶段补充数据：

```text
+ WEAPD rain / snow
+ Kaggle Multi-class Weather Dataset cloudy / rain / shine
```

但第二阶段补充数据必须通过验证：

```text
合入前训练一次
合入后训练一次
只保留能提升验证 macro-F1 的数据源
```

### 13.7 最终筛选结论

当前最可用的数据集是：

```text
WeatherNet-05-18039
Roboflow Weather Classification Dataset
```

最适合补充雨雪的数据集是：

```text
Weather Image Recognition / WEAPD
```

只适合调试流程的数据集是：

```text
Kaggle Multi-class Weather Dataset
Mendeley Multi-class Weather Dataset
```

先不要优先投入的数据集是：

```text
BDD100K / RWVC-BDD100K
DAWN / ACDC
```

执行优先级：

```text
1. 先下载 WeatherNet-05-18039
2. 再下载 Roboflow Weather Classification Dataset
3. 建立四分类统一 CSV
4. 删除 foggy / hazy / sunrise 等非目标类别
5. 抽样检查每类图片质量
6. 训练 baseline
7. 等官方训练集发布后做迁移微调
```
