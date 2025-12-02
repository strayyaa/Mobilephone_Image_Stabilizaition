if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# Configuration
$seq_len = 200
$model_name = "DLinear_XYZ"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "AngleDataDLinearXYZ"
$data_name = "angle"

# Model parameters
$enc_in = 3
$label_len = 48
$moving_avg = 25

# Training parameters (tuned for integrated pipeline)
$train_epochs = 20
$patience = 5
$batch_size = 128
$learning_rate = 0.001
$random_seed = 2021
$itr = 1

$pred_lengths = @(200)
$individual = 1

Write-Host "Starting DLinear_XYZ training..." -ForegroundColor Green

foreach ($pred_len in $pred_lengths) {
    $model_id = "$model_id_name" + "_sl" + "$seq_len" + "_pl" + "$pred_len"
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
      --features M `
      --seq_len $seq_len `
      --label_len $label_len `
      --pred_len $pred_len `
      --enc_in $enc_in `
      --dec_in $enc_in `
      --c_out $enc_in `
      --moving_avg $moving_avg `
      --des 'Exp' `
      --train_epochs $train_epochs `
      --patience $patience `
      --itr $itr `
      --no_use_augmentation `
      --no_use_smoothing `
        --batch_size $batch_size `
        --learning_rate $learning_rate `
        --individual $individual `
        --dlinear_xyz_use_autoformer `
        --dlinear_xyz_top_k_fft 64 `
        --dlinear_xyz_top_k_corr 4 `
        --dlinear_xyz_layers_cnt 1 `
        --dlinear_xyz_layers_cnt_encoder 1 | Tee-Object -FilePath $log_file

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed: $log_file" -ForegroundColor Red
    }
}
