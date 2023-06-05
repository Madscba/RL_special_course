$laptop_paths=False

$test_name = "REINFORCE"
echo "starting $test_name"



if ($laptop_paths) {
echo "Laptop"
$python_executable1 = "C:\Users\Mads-\AppData\Local\pypoetry\Cache\virtualenvs\rl-NjavQPoj-py3.8\Scripts\python.exe"
$repo_path = "C:\Users\Mads-\Documents\Universitet\Kandidat\4_semester\RL"
} else {
$python_executable1 = "C:\Users\Mads-\miniconda3\envs\RL_special_course\python.exe"
$repo_path = "C:\Users\Mads-\Documents\universitet\4_sem\RL"
}

$file_path = "RL_special_course\hand_in\main.py"
$full_path = Join-Path -Path $repo_path -ChildPath $file_path
echo $full_path

& $python_executable1 $full_path --seed 3 --env_name LunarLanderContinuous-v2 --log_format "default" --frame_interval 10000 --visualize 1 --n_steps 400000 --n_env 1 --algorithm SAC_v0 --gamma 0.99 --hidden_size 256 --lr 0.0003 --batch_size 256 --grad_clipping 0 --tau 0.005 --reward_scale 2.0


# Check the exit code of the Python script
if ($LASTEXITCODE -eq 0) {
    Write-Output "$test_name script completed successfully."
} else {
    Write-Output "$test_name script encountered an error with exit code $LASTEXITCODE."
}