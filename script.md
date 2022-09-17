python -u run.py --is_training 1 --root_path ./dataset/ --data_path Exchange.csv --model_id Exchange_96 --model NLinear --data custom --features M --seq_len 96 --pred_len 96 --enc_in 8 --des 'Exp' --itr 1 --batch_size 8 --learning_rate 0.0005 --gpu 1

python -u run.py --is_training 1 --root_path ./dataset/ --data_path Exchange.csv --model_id Exchange_96 --model Informer --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 --train_epochs 1

python -u run.py --is_training 1 --root_path ./dataset/ --data_path Weather.csv --model_id Weather_96 --model NLinear --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --des 'Exp' --itr 1 --batch_size 16

python -u run.py --is_training 1 --root_path ./dataset/ --data_path Weather.csv --model_id Weather_96 --model Informer --data custom --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 --train_epochs 2

python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96 --model NLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.005

python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96 --model Informer --data ETTh1 --features M --seq_len 96 --label_len 1 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1

python -u run.py --is_training 1 --root_path ./dataset/ --data_path ILI.csv --model_id ILI_60 --model NLinear --data custom --features M --seq_len 60 --label_len 1 --pred_len 60 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.01

python -u run.py --is_training 1 --root_path ./dataset/ --data_path ILI.csv --model_id ILI_60 --model Informer --data custom --features M --seq_len 60 --label_len 1 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1