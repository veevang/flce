#!/bin/bash

# 创建日期文件夹
today=$(date +"%Y_%m_%d")
mkdir -p "result/exp_result/$today"
i=0
while true; do
  sub_folder=$(printf "%02d" $i)
  sub_folder_path="result/exp_result/$today/$sub_folder"
  if [ ! -d "$sub_folder_path" ]; then
    mkdir "$sub_folder_path"
    echo "Created folder: $sub_folder_path"
    break
  fi
  i=$((i+1))
done

#num_attack_clients=1
#attack_arg=0.3
#num_attack_clients=2
#num_attack_clients=3

#num_attack_clients=1
#attack_arg=0.1
#attack_arg=0.5
#attack_arg=0.7

# effective in robust setting
#attack_arg=0.8
#num_attack_clients=2

# robust
#methods=('Individual' 'LeaveOneOut' 'ShapleyValue' 'LeastCore' 'MC_StructuredSampling_Shapley' 'MC_LeastCore')
#
## effective
#methods=('RandomMethod' 'Individual' 'LeaveOneOut' 'ShapleyValue' 'LeastCore' 'TMC_Shapley' 'TMC_GuidedSampling_Shapley' 'MC_StructuredSampling_Shapley' 'MC_LeastCore')
#methods=('TMC_Shapley' 'TMC_GuidedSampling_Shapley')
#methods=('Multi_Rounds' 'GTG_Shapley')
#methods=('Individual' 'LeaveOneOut' 'ShapleyValue' 'LeastCore' 'MC_StructuredSampling_Shapley' 'MC_LeastCore' 'RandomMethod')
#methods=('RandomMethod')
#methods=('ShapleyValue')

## efficient
#methods=('Individual' 'LeaveOneOut' 'ShapleyValue' 'LeastCore' 'MC_StructuredSampling_Shapley' 'MC_LeastCore')
#methods=('ShapleyValue' 'LeastCore' 'MC_StructuredSampling_Shapley' 'MC_LeastCore')
#methods=('ShapleyValue' 'LeastCore' 'MC_StructuredSampling_Shapley')
#methods=('MC_LeastCore' 'Individual' 'LeaveOneOut' 'GTG_Shapley')
methods=('Multi_Rounds')

# supplementary experiments: effectiveness 14 nodes
#methods=('MC_StructuredSampling_Shapley' 'MC_LeastCore' 'Individual' 'LeaveOneOut' 'RandomMethod')
#methods=('MC_StructuredSampling_Shapley')
#methods=('MC_LeastCore')
#methods=('Individual')
#methods=('LeaveOneOut')
#methods=('RandomMethod')

#methods=('RandomMethod' 'Individual' 'LeaveOneOut' 'MC_StructuredSampling_Shapley' 'MC_LeastCore')
# supplementary experiments: robust

#methods=('MC_StructuredSampling_Shapley')

# cache for robust and effectiveness
#methods=('ShapleyValue')
#
##robust, effective
#seed=6694
#num_repeat=10
#
##efficient
#seed=6694
#num_repeat=3
#
#value_functions=("f1" "f1_macro" "f1_micro" "accuracy")

#for method in "${methods[@]}"
#do
  # effective quantity
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.365 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.35 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/bank quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.4 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/dota2 quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.65 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/tictactoe quantity skew $method.out" 2>&1 &

  # effective label
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/bank label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/dota2 label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/tictactoe label skew $method.out" 2>&1 &

# [need further implementation] effective urlrep dataset (quantity + label skew)
# params not adjusted
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset urlrep --model UrlMLP --lr 0.005 --num_epoch 60 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/url quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset urlrep --model UrlMLP --lr 0.005 --num_epoch 60 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/url label skew $method.out" 2>&1 &

# effective creditcard dataset (quantity + label skew)
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset creditcard --model CreditCardMLP --lr 0.01 --num_epoch 3 --hidden_layer_size 4 --batch_size 256 --device cpu -a 0.3 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/creditcard quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset creditcard --model CreditCardMLP --lr 0.01 --num_epoch 3 --hidden_layer_size 4 --batch_size 256 --device cpu -a 0.6 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/creditcard label skew $method.out" 2>&1 &


  # effective quantity 14 clients
#  nohup python -u main.py --num_parts 14 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.625 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 14 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.625 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/dota2 quantity skew $method 14.out" 2>&1 &

  # effective label 14 clients
#  nohup python -u main.py --num_parts 14 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 14 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/dota2 label skew $method 14.out" 2>&1 &

#  # robust original
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/bank label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/dota2 label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/tictactoe label skew $method.out" 2>&1 &

  # robust replication
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/adult label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/bank label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust dota2 label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust tictactoe label skew $method data replication.out" 2>&1 &
#
  # robust random generation
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust adult label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust bank label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust dota2 label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust tictactoe label skew $method random data generation.out" 2>&1 &
#
#
#  # robust low quality
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/adult label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/bank label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust dota2 label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust tictactoe label skew $method low quality data.out" 2>&1 &
#
  # robust label flip
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust adult label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust bank label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust dota2 label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/robust tictactoe label skew $method label flip.out" 2>&1 &
#
  # efficient quantity
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.365 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.35 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/bank quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.4 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/efficient dota2 quantity skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 3 -a 0.65 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/efficient tictactoe quantity skew $method.out" 2>&1 &
#
#  # efficient label
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/adult label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/bank label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/efficient dota2 label skew $method.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./result/log/efficient tictactoe quantity skew $method.out" 2>&1 &


#  # effective in robust setting
#
#  # robust replication
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting adult label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting bank label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting dota2 label skew $method data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting tictactoe label skew $method data replication.out" 2>&1 &
#
#
#  # robust random generation
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting adult label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting bank label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting dota2 label skew $method random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting tictactoe label skew $method random data generation.out" 2>&1 &
#
#
#  # robust low quality
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting adult label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting bank label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting dota2 label skew $method low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting tictactoe label skew $method low quality data.out" 2>&1 &
#
#  # robust flip
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting adult label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting bank label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting dota2 label skew $method label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/effective_in_robust_setting tictactoe label skew $method label flip.out" 2>&1 &
#
#done



# data utility
method='ShapleyValue'
value_functions=('accuracy')
#value_functions=('gradient_similarity')
#value_functions=('data_quantity')
#value_functions=('gradient_similarity' 'data_quantity')
# robust volume only for tictactoe
#value_functions=('robust_volume')

#seed=6694
#num_repeat=10

# efficient
seed=6694
num_repeat=3

# effective_in_robust_setting!
#num_attack_clients=2
#attack_arg=0.8

# robust
#num_attack_clients=1
#attack_arg=0.3

for value_function in "${value_functions[@]}"
do
#   effective quantity
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.365 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/adult quantity skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.35 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/bank quantity skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.4 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility dota2 quantity skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 3 -a 0.65 --distribution "quantity skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility tictactoe quantity skew $method $value_function.out" 2>&1 &
#
#   effective label
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/adult label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/bank label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility dota2 label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility tictactoe label skew $method $value_function.out" 2>&1 &

##  effective in robust setting replication
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function data replication.out" 2>&1 &
#
#
##  effective robust setting random generation
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function random data generation.out" 2>&1 &
#
#
##  effective robust setting low quality
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function low quality data.out" 2>&1 &
#
##  effective robust setting flip
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t effective_in_robust_setting -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function label flip.out" 2>&1 &

#  robustness replication
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function data replication.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "data replication" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function data replication.out" 2>&1 &


#  robustness random generation
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function random data generation.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "random data generation" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function random data generation.out" 2>&1 &


#  robustness low quality
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function low quality data.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "low quality data" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function low quality data.out" 2>&1 &

#  robustness flip
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust adult label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust bank label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust dota2 label skew $method $value_function label flip.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t robust -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function --attack "label flip" --attack_arg $attack_arg --num_attack_clients $num_attack_clients > "./result/log/utility robust tictactoe label skew $method $value_function label flip.out" 2>&1 &


#   efficient label
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset adult --model AdultMLP --lr 0.001 --num_epoch 25 --hidden_layer_size 24 --batch_size 64 --device 0 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/adult label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset bank --model BankMLP --lr 0.001 --num_epoch 20 --hidden_layer_size 8 --batch_size 64 --device 1 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/bank label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset dota2 --model Dota2MLP --lr 0.001 --num_epoch 5 --hidden_layer_size 4 --batch_size 128 --device 2 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility dota2 label skew $method $value_function.out" 2>&1 &
#  nohup python -u main.py --num_parts 8 -t efficient -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.008 --num_epoch 80 --hidden_layer_size 16 --batch_size 16 --device 3 -a 0.8 --distribution "label skew" -s $seed --num_repeat $num_repeat --start_date $today --num_try $sub_folder --value_functions $value_function > "./result/log/utility tictactoe label skew $method $value_function.out" 2>&1 &

done