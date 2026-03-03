if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# Configuration
$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "imu_data_with_jitter_reduced.xlsx"
$model_id_name = "IMUDataMoE"
$data_name = "imu"
$target = "yaw"  # 主要预测目标列

# Model parameters
$enc_in = 3
$d_model = 64
$d_ff = 128
$e_layers = 2
$n_heads = 2
$dropout = 0.3
$fc_dropout = 0.1
$head_dropout = 0.0

# PatchTST parameters
$patch_len = 16
$stride = 1

# Training parameters
$train_epochs = 10
$patience = 5
$batch_size = 32
$learning_rate = 0.0001
$random_seed = 2021

$pred_lengths = @(200)
$jitter_sigma = 0.08
$scale_alpha = 0.1
$individual = 1
$downsample_rate = 10

$auxi_lambda = 0
$rec_lambda = 1

# MoE enhancement parameters
$moe_low_freq_ratio = 0.005
$moe_high_freq_ratio = 0.01
$moe_cutoff_ratio = 0.02
$moe_freq_residual = 0.7
$moe_modal_residual = 0.7
$moe_temperature = 0.85
$moe_frequency_experts = 9
$moe_frequency_topk = 3
$moe_energy_norm = "density"

Write-Host "Starting PatchTST training with MoE-enhanced inputs on IMU dataset..." -ForegroundColor Green

foreach ($pred_len in $pred_lengths) {
    $model_id = "$model_id_name" + "_" + "$seq_len" + "_" + "$pred_len"
    $log_file = "logs/LongForecasting/$model_name" + "_" + "$model_id.log"

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $py = 'python'
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $py = 'python3'
    } else {
        Write-Host "未找到 python 可执行，请安装 Python 并确保 'python' 或 'python3' 在 PATH 中" -ForegroundColor Red
        exit 1
    }

    & $py -u ../../run_longExp.py `
      --random_seed $random_seed `
      --is_training 1 `
      --root_path $root_path_name `
      --data_path $data_path_name `
      --model_id $model_id `
      --model $model_name `
      --data $data_name `
      --use_moe `
      --moe_low_freq_ratio $moe_low_freq_ratio `
      --moe_high_freq_ratio $moe_high_freq_ratio `
      --moe_cutoff_ratio $moe_cutoff_ratio `
      --moe_frequency_residual_ratio $moe_freq_residual `
      --moe_modal_residual_ratio $moe_modal_residual `
      --moe_temperature $moe_temperature `
      --moe_frequency_experts $moe_frequency_experts `
      --moe_frequency_topk $moe_frequency_topk `
      --moe_energy_norm $moe_energy_norm `
      --features M `
      --seq_len $seq_len `
      --label_len 48 `
      --pred_len $pred_len `
      --enc_in $enc_in `
      --dec_in $enc_in `
      --c_out $enc_in `
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
      --des 'ExpMoE' `
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
      --check_self_correlation `
      --auxi_lambda $auxi_lambda `
      --rec_lambda $rec_lambda `
      --target $target `
      --output_data_print `
      --self_corr_channel 2 `
            2>&1 | Tee-Object -FilePath $log_file

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ MoE training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ MoE training failed: $log_file" -ForegroundColor Red
    }
}
