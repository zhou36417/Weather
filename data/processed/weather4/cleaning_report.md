# WeatherNet 四分类清洗报告

## Summary

- Source rows: 18039
- Kept images: 16562
- Train images: 13249
- Val images: 3313
- Validation ratio: 0.2

## Source Label Counts

| Source Label | Name | Count | Action |
|---:|---|---:|---|
| 0 | cloudy or overcast | 6702 | map to cloudy |
| 1 | foggy or hazy | 1261 | drop |
| 2 | rain or storm | 1927 | map to rainy |
| 3 | snow or frosty | 1875 | map to snowy |
| 4 | sun or clear | 6274 | map to sunny |

## Target Counts

| Target | Count |
|---|---:|
| cloudy | 6515 |
| rainy | 1925 |
| snowy | 1875 |
| sunny | 6247 |

## Drop Counts

| Reason | Count |
|---|---:|
| exact_duplicate | 216 |
| non_target_label | 1261 |

## Outputs

- `metadata.csv`: all kept cleaned images
- `train.csv`: stratified training split
- `val.csv`: stratified validation split
- `label_map.json`: unified four-class label mapping
- `reports/samples/*.jpg`: sample contact sheets for visual QA