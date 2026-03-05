# Optimization: 1 expert with longer training and stronger augmentation
# Hypothesis: More training epochs + stronger augmentation may improve results

$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "PhoneData_FreqMoE_1Expert_Opt"
$data_name = "angle"
$target = "x_angle_1"

$enc_in = 3
$d_model = 64
$d_ff = 128
$e_layers = 2
$n_heads = 2
$dropout = 0.3
$fc_dropout = 0.1
$head_dropout = 0.0

$patch_len = 16
$stride = 1

# Key optimization: more epochs + stronger augmentation
$train_epochs = 12  # More epochs
$patience = 7       # More patience
$batch_size = 32
$learning_rate = 0.0001
$random_seed = 2021

$pred_len = 200
# Stronger augmentation
$jitter_sigma = 0.12  # Increased from 0.08
$scale_alpha = 0.15   # Increased from 0.1
$individual = 1
$downsample_rate = 10

$auxi_lambda = 0
$rec_lambda = 1

# 1 expert (FFT normalization)
$freqmoe_experts = 1
$freqmoe_use_norm = 1
$freqmoe_use_soft_mask = 0
$freqmoe_sharpness = 30.0
$freqmoe_freq_wise_routing = 0
$freqmoe_balance_weight = 0.0

$model_id = $model_id_name + "_Ep" + $train_epochs + "_Jit" + ($jitter_sigma * 100) + "_" + $seq_len + "_" + $pred_len

Write-Host "Running optimized 1-expert (more epochs + stronger augmentation)..." -ForegroundColor Green

python -u ../../run_longExp.py `
  --random_seed $random_seed `
  --is_training 1 `
  --root_path $root_path_name `
  --data_path $data_path_name `
  --model_id $model_id `
  --model $model_name `
  --data $data_name `
  --target $target `
  --use_freqmoe `
  --freqmoe_experts $freqmoe_experts `
  --freqmoe_use_norm $freqmoe_use_norm `
  --freqmoe_use_soft_mask $freqmoe_use_soft_mask `
  --freqmoe_sharpness $freqmoe_sharpness `
  --freqmoe_freq_wise_routing $freqmoe_freq_wise_routing `
  --freqmoe_balance_weight $freqmoe_balance_weight `
  --features M `
  --seq_len $seq_len `
  --label_len 48 `
  --pred_len $pred_len `
  --enc_in $enc_in `
  --dec_in $enc_in `
  --c_out 3 `
  --e_layers $e_layers `
  --d_layers 1 `
  --n_heads $n_heads `
  --d_model $d_model `
  --d_ff $d_ff `
  --dropout $dropout `
  --fc_dropout $fc_dropout `
  --head_dropout $head_dropout `
  --patch_len $patch_len `
  --stride $stride `
  --train_epochs $train_epochs `
  --patience $patience `
  --itr 1 `
  --batch_size $batch_size `
  --num_workers 0 `
  --learning_rate $learning_rate `
  --use_augmentation `
  --use_smoothing `
  --jitter_sigma $jitter_sigma `
  --scale_alpha $scale_alpha `
  --individual $individual `
  --downsample_rate $downsample_rate `
  --auxi_lambda $auxi_lambda `
  --rec_lambda $rec_lambda
