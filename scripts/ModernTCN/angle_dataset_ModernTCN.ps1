if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# Configuration
$seq_len = 200
$model_name = "ModernTCN"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "AngleDataModernTCN"
$data_name = "angle"

# ModernTCN hyperparameters
$enc_in = 3
$label_len = 48
$stem_ratio = 6
$downsample_ratio = 2
$ffn_ratio = 2
$patch_size = 16
$patch_stride = 8
$num_blocks = @(1,1,1,1)
$large_size = @(31,29,27,13)
$small_size = @(5,5,5,5)
$dims = @(128,128,128,128)
$use_multi_scale = 1
$revin = 1
$affine = 0
$subtract_last = 0
$decomposition = 0
$head_dropout = 0.1
$dropout = 0.2

# Training setup
$train_epochs = 15
$patience = 5
$batch_size = 16
$learning_rate = 0.0003
$random_seed = 2021
$itr = 1

$pred_lengths = @(200)
$jitter_sigma = 0.08
$scale_alpha = 0.1
$downsample_rate = 10

$auxi_lambda = 0
$rec_lambda = 1

Write-Host "Starting ModernTCN training..." -ForegroundColor Green

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
      --freq s `
      --seq_len $seq_len `
      --label_len $label_len `
      --pred_len $pred_len `
      --enc_in $enc_in `
      --dec_in $enc_in `
      --c_out $enc_in `
      --stem_ratio $stem_ratio `
      --downsample_ratio $downsample_ratio `
      --ffn_ratio $ffn_ratio `
      --patch_size $patch_size `
      --patch_stride $patch_stride `
      --num_blocks $num_blocks `
      --large_size $large_size `
      --small_size $small_size `
      --dims $dims `
      --use_multi_scale $use_multi_scale `
      --revin $revin `
      --affine $affine `
      --subtract_last $subtract_last `
      --decomposition $decomposition `
      --head_dropout $head_dropout `
      --dropout $dropout `
      --des 'Exp' `
      --train_epochs $train_epochs `
      --patience $patience `
      --itr $itr `
      --batch_size $batch_size `
      --learning_rate $learning_rate `
      --use_augmentation `
      --use_smoothing `
      --downsample_rate $downsample_rate `
      --jitter_sigma $jitter_sigma `
      --scale_alpha $scale_alpha `
      --auxi_lambda $auxi_lambda `
      --rec_lambda $rec_lambda | Tee-Object -FilePath $log_file

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed: $log_file" -ForegroundColor Red
    }
}
