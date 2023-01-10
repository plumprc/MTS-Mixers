<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 321 --itr 1 --learning_rate 0.001 --model MTSMixer --d_ff 16 --train_epochs 6 --rev --norm --fac_C --refine

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --itr 1 --model Transformer --train_epochs 5 --rev

<!-- Traffic -->

python -u run.py --is_training 1 --data_path Traffic.csv --data custom --features M --seq_len 96 --pred_len 96 --e_layers 1 --enc_in 862 --itr 1 --learning_rate 0.05 --model MTSMixer --d_ff 64 --rev --fac_C --train_epochs 3

python -u run.py --is_training 1 --data_path Traffic.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --itr 1 --model Transformer --learning_rate 0.05 --train_epochs 4 --rev

<!-- PeMS04 -->

python -u run.py --is_training 1 --data_path PeMS04.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 921 --itr 1 --learning_rate 0.005 --model MTSMixer --d_ff 64 --fac_C --train_epochs 1 --refine

python -u run.py --is_training 1 --data_path PeMS04.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 921 --dec_in 921 --c_out 921 --itr 1 --model Transformer --train_epochs 4 --rev

<!-- Exchange -->

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 8 --itr 1 --batch_size 8 --learning_rate 0.0005 --model MTSMixer --rev --norm --d_model 256 --fac_T --train_epochs 3

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --itr 1 --model Transformer --train_epochs 1 --rev

<!-- Weather -->

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --itr 1 --model MTSMixer --rev --norm --d_model 1024 --train_epochs 6 --refine

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --itr 1 --model Transformer --train_epochs 1 --rev

<!-- ILI -->

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 24 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.01 --model MTSMixer --rev --norm --fac_T

python -u run.py --is_training 1 --data_path ILI.csv --data custom --features M --seq_len 36 --label_len 1 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev

---

<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --rev --norm --train_epochs 6 --fac_T --sampling 6

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev

<!-- ETTm1 -->

python -u run.py --is_training 1 --data_path ETTm1.csv --data ETTm1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --train_epochs 6 --rev --norm --fac_T --sampling 8

python -u run.py --is_training 1 --data_path ETTm1.csv --data ETTm1 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev

<!-- ETTh2 -->

python -u run.py --is_training 1 --data_path ETTh2.csv --data ETTh2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --rev --norm --train_epochs 6

python -u run.py --is_training 1 --data_path ETTh2.csv --data ETTh2 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev

<!-- ETTm2 -->

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model MTSMixer --train_epochs 6 --rev --norm --fac_T --sampling 3

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --itr 1 --model Transformer --train_epochs 1 --rev