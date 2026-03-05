# freqMOE with Fine-Grained Low-Frequency Experts
# Idea: Instead of spreading experts across full frequency range (0-100%),
# concentrate experts in the low-frequency portion (e.g., 0-30%) with finer granularity
#
# Example:
# - Original: Expert1=[0-25%], Expert2=[25-50%], Expert3=[50-75%], Expert4=[75-100%]
# - New:      Expert1=[0-7.5%], Expert2=[7.5-15%], Expert3=[15-22.5%], Expert4=[22.5-30%]

param(
    [int]$expert_freq_range = 30,  # Expert frequency range percentage (default 30%)
    [int]$num_experts = 4          # Number of experts
)

$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "PhoneData_FreqMoE_FineGrained"
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

# freqMoE with fine-grained low-frequency experts
$freqmoe_experts = $num_experts
$freqmoe_use_norm = 1
$freqmoe_use_soft_mask = 0  # Use hard mask for clearer frequency boundaries
$freqmoe_sharpness = 30.0
$freqmoe_freq_wise_routing = 0
$freqmoe_balance_weight = 0.0
$freqmoe_use_lowpass_filter = 0  # Don't filter, just focus experts in low freq range
$freqmoe_expert_freq_range = $expert_freq_range / 100.0  # Convert to ratio

$model_id = $model_id_name + "_Range" + $expert_freq_range + "_Experts" + $num_experts + "_" + $seq_len + "_" + $pred_len

Write-Host "Running freqMOE with Fine-Grained Low-Frequency Experts (Range=$expert_freq_range%, Experts=$num_experts)..." -ForegroundColor Green

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
