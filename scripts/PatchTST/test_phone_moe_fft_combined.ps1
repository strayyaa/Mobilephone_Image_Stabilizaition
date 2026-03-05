# Optimization: Combine MoE + FFT Enhancement
# Hypothesis: MoE (0.0939) + FFT Enhancement (0.0932) may work better together

$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "PhoneData_MoE_FFT_Combined"
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

$train_epochs = 8
$patience = 5
$batch_size = 32
$learning_rate = 0.0001
$random_seed = 2021

$pred_len = 200
$jitter_sigma = 0.08
$scale_alpha = 0.1
$individual = 1
$downsample_rate = 10

# Enable BOTH MoE AND FFT Enhancement
$use_moe = $true
$moe_frequency_experts = 9
$moe_frequency_topk = 3

$use_fft_enhance = $true
$fft_low_freq_ratio = 0.25
$fft_high_freq_ratio = 0.7
$fft_low_freq_boost = 1.3
$fft_high_freq_suppress = 0.2

$model_id = $model_id_name + "_" + $seq_len + "_" + $pred_len

Write-Host "Running MoE + FFT Enhancement combined..." -ForegroundColor Green

python -u ../../run_longExp.py `
  --random_seed $random_seed `
  --is_training 1 `
  --root_path $root_path_name `
  --data_path $data_path_name `
  --model_id $model_id `
  --model $model_name `
  --data $data_name `
  --target $target `
  --use_moe `
  --moe_frequency_experts $moe_frequency_experts `
  --moe_frequency_topk $moe_frequency_topk `
  --use_fft_enhance_data `
  --fft_low_freq_ratio $fft_low_freq_ratio `
  --fft_high_freq_ratio $fft_high_freq_ratio `
  --fft_low_freq_boost $fft_low_freq_boost `
  --fft_high_freq_suppress $fft_high_freq_suppress `
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
