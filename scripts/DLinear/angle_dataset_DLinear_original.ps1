if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# 基础配置
$seq_len = 200
$model_name = "DLinear"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "AngleData"
$data_name = "angle"
$target = "z_angle_1"

# 训练超参
$train_epochs = 10
$patience = 5
$batch_size = 32
$learning_rate = 0.0001
$random_seed = 2021
$itr = 1

$pred_lengths = @(200)

Write-Host "Starting original DLinear training..." -ForegroundColor Green

foreach ($pred_len in $pred_lengths) {
    $model_id = "{0}_{1}_{2}" -f $model_id_name, $seq_len, $pred_len
    $log_file = "logs/LongForecasting/{0}_{1}.log" -f $model_name, $model_id

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
      --label_len 48 `
      --pred_len $pred_len `
      --enc_in 3 `
      --dec_in 3 `
      --c_out 3 `
      --des 'Exp' `
      --train_epochs $train_epochs `
      --patience $patience `
      --itr $itr `
      --batch_size $batch_size `
      --learning_rate $learning_rate `
      --use_augmentation `
      --use_smoothing `
      --check_self_correlation `
      --target $target `
      2>&1 | Tee-Object -FilePath $log_file

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed: $log_file" -ForegroundColor Red
    }
}
