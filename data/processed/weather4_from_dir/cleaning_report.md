# 目录型天气四分类清洗报告

## Summary

- Input root: `C:/Users/Z Pengfei/Desktop/caip/data/data`
- Kept images: 16533
- Train images: 13226
- Val images: 3307
- Validation ratio: 0.2

## Class Counts

| Source Class | Count | Action |
|---|---:|---|
| cloudy | 6702 | map to cloudy |
| foggy | 1261 | drop |
| rainy | 1927 | map to rainy |
| snowy | 1875 | map to snowy |
| sunny | 6274 | map to sunny |

## Strict Target Counts

| Target | Total | Train | Val |
|---|---:|---:|---:|
| cloudy | 6486 | 5189 | 1297 |
| rainy | 1925 | 1540 | 385 |
| snowy | 1875 | 1500 | 375 |
| sunny | 6247 | 4997 | 1250 |

## Drop Counts

| Reason | Count |
|---|---:|
| dropped_class:foggy | 1261 |
| duplicate_label_conflict | 59 |
| exact_duplicate | 186 |

## Recommended Training Files

- `train_strict.csv`
- `val_strict.csv`
- `metadata_strict.csv`
- `label_map.json`