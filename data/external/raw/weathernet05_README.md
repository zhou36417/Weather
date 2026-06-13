---
license: apache-2.0
task_categories:
- image-classification
language:
- en
tags:
- Weather
- Classification
size_categories:
- 10K<n<100K
---

# WeatherNet-05-18039 

## Overview

WeatherNet-05 is a weather image classification dataset consisting of 18,039 images labeled into 5 distinct weather-related classes. The dataset is suitable for training and evaluating computer vision models on the task of classifying weather conditions based on image data.

## Dataset Structure

- **Split:** `train`  
- **Number of rows:** 18,039  
- **Label Type:** Categorical (5 classes)  
- **Image Resolution:** Varies (from 90px to 4.86k px width)  
- **File Format:** Auto-converted to Parquet for efficient processing

## Label Classes

The dataset contains the following classes (not fully visible in the image but inferred from partial data):

- cloudy or overcast
- [4 other class names not displayed in the screenshot]

## Usage

You can use the dataset directly with Hugging Face's `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("prithivMLmods/WeatherNet-05-18039")
````

## Applications

This dataset is ideal for:

* Weather image classification
* Transfer learning with visual transformers
* Fine-tuning pre-trained computer vision models

## Related Models

This dataset has been used to train or fine-tune models such as:

* `prithivMLmods/Weather-Image-Classification` (Image Classification)

## Collections

This dataset is part of the collection:

* `Content Filters SigLIP2/ViT` (Moderation, Balance, Contextual Understanding)