# MTSformer
MTSformer: towards universal multivariate time series modeling based on Transformer?

## Baseline
### Exchange 96-1-96

|Model|MSE|MAE
|:-:|:-:|:-:
|Linear|0.084|0.210
|ConvFC|0.081|0.207
|FEDformer|0.148|0.278
|MTS-Mixer (d_model=512) + RevIN|0.134|0.258
|MTS-Mixer (d_model=4) + RevIN|0.115|0.243
|MTSMatrix (d_model=16)|0.083|0.202

### ETTm2 96-1-96

|Model|MSE|MAE
|:-:|:-:|:-:
|Linear|0.183|0.275
|ConvFC|0.184|0.276

### Weather 96-1-96

|Model|MSE|MAE
|:-:|:-:|:-:
|Linear|0.199|0.258
|ConvFC|Weather|0.199|0.257
|FEDformer|Weather|0.217|0.296
|MTS-Mixer (d_model=512) + RevIN|Weather|0.167|0.221
|MTSMatrix (d_model=512)|0.167|0.216

## Normalization
### Exchange 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Transformer|0.611|0.616
|Transformer + short_cut|0.446|0.527
|Transformer + ModifiedLN|0.626|0.628
|Transformer + RevIN|0.157|0.286

### ETTm2 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Transformer|0.428|0.482
|Transformer + short_cut|0.486|0.523
|Transformer + ModifiedLN|0.425|0.479
|Transformer + RevIN|0.229|0.302

### Weather 96-1-96

|model|MSE|MAE
|:-:|:-:|:-:
|Transformer|0.363|0.426
|Transformer + short_cut|0.478|0.502
|Transformer + ModifiedLN|0.347|0.412
|Transformer + RevIN|0.181|0.230