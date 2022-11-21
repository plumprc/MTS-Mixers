<!-- Exchange -->

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 8 --itr 1 --batch_size 8 --learning_rate 0.0005 --model Linear --gpu 3

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --itr 1 --model Transformer --train_epochs 1

<!-- Weather -->

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --itr 1 --batch_size 16 --model NLinear

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --itr 1 --model Transformer --train_epochs 2

<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model NLinear

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1

<!-- ETTm2 -->

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model NLinear

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1

<!-- ILI -->

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 60 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.01 --model NLinear

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1

<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 321 --itr 1 --batch_size 16  --learning_rate 0.001 --model NLinear

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --itr 1 --model Transformer --train_epochs 1

<!-- Traffic -->

python -u run.py --is_training 1 --data_path Traffic.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 862 --itr 1 --batch_size 16 --learning_rate 0.05 --model SCINet
