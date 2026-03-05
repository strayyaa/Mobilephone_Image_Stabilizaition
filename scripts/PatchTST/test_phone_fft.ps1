# Test without any freq processing - to verify FFT normalization effect
$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "PhoneData_FFT"
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

$auxi_lambda = 0
$rec_lambda = 1

# Use FFT enhance instead of freqMOE
$fft_low_freq_ratio = 0.25
$fft_high_freq_ratio = 0.7
$fft_cutoff_ratio = 0.9
$fft_low_freq_boost = 1.3
$fft_mid_freq_boost = 1.1
$fft_high_freq_suppress = 0.2
$fft_residual_ratio = 0.5

$model_id = $model_id_name + "_" + $seq_len + "_" + $pred_len

Write-Host "Running FFT enhance (not freqMOE)..." -ForegroundColor Green

python -u ../../run_longExp.py `
  --random_seed $random_seed `
  --is_training 1 `
  --root_path $root_path_name `
  --data_path $data_path_name `
  --model_id $model_id `
  --model $model_name `
  --data $data_name `
  --use_fft_enhance_data `
  --fft_low_freq_ratio $fft_low_freq_ratio `
  --fft_high_freq_ratio $fft_high_freq_ratio `
  --fft_cutoff_ratio $fft_cutoff_ratio `
  --fft_low_freq_boost $fft_low_freq_boost `
  --fft_mid_freq_boost $fft_mid_freq_boost `
  --fft_high_freq_suppress $fft_high_freq_suppress `
  --fft_residual_ratio $fft_residual_ratio `
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
  --des 'FFT_Phone' `
  --train_epochs $train_epochs `
  --patience $patience `
  --itr 1 `
  --batch_size $batch_size `
  --num_workers 0 `
  --learning_rate $learning_rate `
  --use_augmentation `
  --use_smoothing `
  --downsample_rate $downsample_rate `
  --jitter_sigma $jitter_sigma `
  --scale_alpha $scale_alpha `
  --individual $individual `
  --auxi_lambda $auxi_lambda `
  --rec_lambda $rec_lambda `
  --target $target

Write-Host "FFT enhance completed!" -ForegroundColor Green
