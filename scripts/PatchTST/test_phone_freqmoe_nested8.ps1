# freqMOE with Nested Fine-Grained Experts
# Idea: Use MORE experts (8) in a small low-frequency range (20%)
# This creates finer granularity - each expert covers ~2.5% of frequency range
# Expected: More experts should allow better specialization in low-freq region

param(
    [int]$expert_freq_range = 20,  # Expert frequency range percentage
    [int]$num_experts = 8,         # MORE experts = finer granularity!
    [int]$sharpness = 50            # Higher sharpness for clearer boundaries
)

$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "PhoneData_FreqMoE_Nested8Experts"
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

$train_epochs = 3
$patience = 5
$batch_size = 32
$learning_rate = 0.0001
$random_seed = 2021

$pred_len = 200
$jitter_sigma = 0.08
$scale_alpha = 0.1
$individual = 1
$downsample_rate = 10

# Nested fine-grained experts: 8 experts in 20% range = ~2.5% per expert
$freqmoe_experts = $num_experts
$freqmoe_use_norm = 1
$freqmoe_use_soft_mask = 1  # Soft mask for learnable boundaries
$freqmoe_sharpness = $sharpness
$freqmoe_freq_wise_routing = 0
$freqmoe_balance_weight = 0.02  # Slightly higher balance loss for more experts
$freqmoe_use_lowpass_filter = 0
$freqmoe_expert_freq_range = $expert_freq_range / 100.0

$model_id = $model_id_name + "_Range" + $expert_freq_range + "_Experts" + $num_experts + "_Sharp" + $sharpness + "_" + $seq_len + "_" + $pred_len

Write-Host "Running Nested freqMOE ($num_experts experts in $expert_freq_range% range)..." -ForegroundColor Green

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
  --freqmoe_use_lowpass_filter $freqmoe_use_lowpass_filter `
  --freqmoe_expert_freq_range $freqmoe_expert_freq_range `
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
  --downsample_rate $downsample_rate
