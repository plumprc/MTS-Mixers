# MTSformer
MTSformer: towards universal multivariate time series modeling based on Transformer?

## Baseline
### Transformer-based models (96-96)

|Dataset|MSE|MAE
|:-:|:-:|:-:
|ETTh1|0.499|0.484
|Weather|0.192|0.243
|ECL|0.175|0.280
|ECL_shuffle|0.179|0.285
|ECL_shuffle_min_max|0.176|0.281
|ECL_shuffle_max_min|0.178|0.285
|Traffic|0.686|0.394

### Linear (96-96)

|Dataset|MSE|MAE
|:-:|:-:|:-:
|ETTh1|0.399|0.416
|Weather|0.199|0.258
|ECL|0.195|0.278
|ECL_shuffle|0.195|0.278
|ECL_shuffle_min_max|0.195|0.278
|ECL_shuffle_max_min|0.195|0.278
|Traffic|0.650|0.397

### MTS-Mixer on ECL (96-96)

|Setting|MSE|MAE
|64-32|0.162|0.266
|512-128|0.162|0.267
|512-64|0.159|0.264
|512-32|0.158|0.263
|512-16|0.155|0.259