<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 321 --itr 1 --batch_size 16 --learning_rate 0.001 --model MTSMixer --d_ff 16 --train_epochs 7

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --itr 1 --model Transformer --train_epochs 5 --rev 1

<!-- ETTm2 -->

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --d_model 8 --d_ff 4 --train_epochs 1

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1

<!-- Traffic -->

python -u run.py --is_training 1 --data_path Traffic.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 862 --itr 1 --batch_size 16 --learning_rate 0.05 --model MTSMixer --d_ff 64 --train_epochs 2

python -u run.py --is_training 1 --data_path Traffic.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --itr 1 --model Transformer --learning_rate 0.05 --train_epochs 4 --rev 1

<!-- PeMS04 -->

python -u run.py --is_training 1 --data_path PeMS04.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 921 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --d_ff 64

python -u run.py --is_training 1 --data_path PeMS04.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 921 --dec_in 921 --c_out 921 --itr 1 --model Transformer --train_epochs 4 --rev 1

<!-- Exchange -->

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 8 --itr 1 --batch_size 8 --learning_rate 0.0005 --model Linear --gpu 3

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --itr 1 --model Transformer --train_epochs 1 --rev 1

<!-- Weather -->

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --itr 1 --batch_size 16 --model Linear --d_ff 32

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --itr 1 --model Transformer --train_epochs 1 --rev 1

<!-- ILI -->

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 60 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.01 --model Linear

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1

---

<!-- Appliance -->

python -u run.py --is_training 1 --data_path Appliance.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 28 --itr 1 --batch_size 32 --learning_rate 0.005 --model Linear

python -u run.py --is_training 1 --data_path Appliance.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 28 --dec_in 28 --c_out 28 --itr 1 --model Transformer --train_epochs 4 --rev 1

<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model Transformer_lite --rev 1 --train_epochs 2

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev 1

<!-- TSM -->

python -u run.py --is_training 1 --data_path TSM1.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 4 --dec_in 4 --c_out 4 --itr 1 --model Transformer --train_epochs 1

python -u run.py --is_training 1 --data_path TSM2.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 2 --dec_in 2 --c_out 2 --itr 1 --model Transformer --train_epochs 1