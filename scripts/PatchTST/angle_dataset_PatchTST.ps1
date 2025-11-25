if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# Configuration
$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "AngleData"
$data_name = "angle"
$target = "z_angle_1"  # 主要的预测目标列

# Model parameters (optimized)
$enc_in = 3              # 输入特征数量
$d_model = 64           # 每个token的嵌入的维度
$d_ff = 128             # 每个encoder层的前馈神经网络的维度，每个encoder层 d_model--> d_ff --> d_model
$e_layers = 2            # encoders层数
$n_heads = 2             # 头数
$dropout = 0.3    # Dropout率
$fc_dropout = 0.1         # 全连接层Dropout率
$head_dropout = 0.0        # 头部Dropout率

# PatchTST parameters (optimized)
$patch_len = 16
$stride = 1

# Training parameters (optimized)
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

Write-Host "Starting PatchTST training..." -ForegroundColor Green

foreach ($pred_len in $pred_lengths) {
    $model_id = "$model_id_name" + "_" + "$seq_len" + "_" + "$pred_len"
    $log_file = "logs/LongForecasting/$model_name" + "_" + "$model_id.log"

    # 兼容 macOS / Linux: 自动选择 python 或 python3 可执行文件，并使用 POSIX 风格相对路径
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
      --des 'Exp' `
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
            --check_self_correlation `
            --self_corr_channel 2 `
            2>&1 | Tee-Object -FilePath $log_file


    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed: $log_file" -ForegroundColor Red
    }
}