# MTSformer
MTSformer: towards universal multivariate time series modeling based on Transformer

## Official compared baseline

|Model|Dataset (96-96)|MSE|MAE
|:-:|:-:|:-:|:-:
|Linear|Exchange|0.082|0.207
|FEDformer|Exchange|0.148|0.278
|Linear|ETTh1|0.375|0.397
|FEDformer|ETTh1|0.376|0.419
|Linear|Weather|0.176|0.236
|FEDformer|Weather|0.217|0.296

## Non-Transformer results
### Modified Linear model

|Model|Dataset|MSE|MAE
|:-:|:-:|:-:|:-:
|Linear|Exchange|0.084|0.210
|Linear + RevIN|Exchange|0.083|0.201
|Linear + Fixed PE|Exchange|0.166|0.309
|Linear + Conv (d_model=64)|Exchange|0.172|0.317
|Linear + Linear (d_model=64)|Exchange|0.175|0.317
|Linear + Fixed PE + Conv (d_model=8)|Exchange|0.420|0.491
|Linear + Fixed PE + Conv (d_model=64)|Exchange|0.173|0.316
|Linear + Fixed PE + Conv (d_model=128)|Exchange|0.176|0.313
|Linear + Fixed PE + Conv (d_model=256)|Exchange|0.169|0.313
|Linear + Fixed PE + Conv (d_model=512)|Exchange|0.190|0.335
|Linear + Fixed PE + Conv (d_model=1024)|Exchange|0.182|0.321
|Linear + Fixed PE + Linear (d_model=64)|Exchange|0.178|0.320
|Linear + Fixed PE + Linear (d_model=512)|Exchange|0.164|0.309
|Linear|ETTh1|0.400|0.415
|Linear + Fixed PE|ETTh1|0.408|0.423
|Linear + Fixed PE + Conv (d_model=512)|ETTh1|0.442|0.446

### MLP-style

|Model|Dataset|MSE|MAE
|:-:|:-:|:-:|:-:
|SCINet (d_model=64)|Exchange|1.419|0.919
|SCINet (d_model=64) + RevIN|Exchange|0.142|0.249
|SCINet (d_model=512)|Exchange|0.280|0.394
|SCINet (d_model=512) + RevIN|Exchange|0.114|0.237
|SCINet (d_model=1024)|Exchange|0.234|0.361
|SCINet (d_model=1024) + RevIN|Exchange|0.109|0.235
|MLP (d_model=512)|Exchange|0.389|0.503
|MLP (d_model=512) + RevIN|Exchange|0.089|0.211
|ConvFC|Exchange|0.081|0.207
|ConvFC + RevIN|Exchange|0.083|0.201
|ConvFC|ETTh1|0.400|0.417
|ConvFC + RevIN|ETTh1|0.399|0.405
|ConvFC|Weather|0.199|0.257
|ConvFC + RevIN|Weather|0.197|0.236

## Input Embedding
### Exchange 96-1-96

|Model|Positional Encoding|MSE|MAE
|:-:|:-:|:-:|:-:
|Informer|Default|1.031|0.829
|Transformer|Default|0.611|0.616
|Informer|Fixed PE|1.028|0.829
|Transformer|Fixed PE|0.694|0.659
|Informer|RoPE||
|Transformer|RoPE||
|Informer|Learnable PE||
|Transformer|Learnable PE||

### ETTh1 96-1-96

|Model|Positional Encoding|MSE|MAE
|:-:|:-:|:-:|:-:
|Informer|Default|0.821|0.704
|Transformer|Default|0.811|0.714
|Informer|Fixed PE|0.780|0.682
|Transformer|Fixed PE|0.819|0.716
|Informer|RoPE||
|Transformer|RoPE||
|Informer|Learnable PE||
|Transformer|Learnable PE||

## Temporal/Causal Attention

## A high-level semantic Decoders
### Semantic queries

### Output Normalization
Exchange 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Informer|1.031|0.829
|Informer + RevIN|0.121|0.256
|Informer + Alignment|0.090|0.211
|Transformer|0.611|0.616
|Transformer + RevIN|0.157|0.286
|Transformer + Alignment|0.131|0.276

ETTh1 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Informer|0.821|0.704
|Informer + RevIN|0.616|0.540
|Informer + Alignment|2.168|0.858
|Transformer|0.811|0.714
|Transformer + RevIN|0.534|0.501

Weather 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Informer|0.630|0.581
|Informer + RevIN|0.200|0.246
|Informer + Alignment|0.260|0.266
|Transformer|0.363|0.426
|Transformer + RevIN|0.181|0.230
|Transformer + Alignment|0.380|0.405