$laptop_paths=False

$test_name = "DDQN"
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

& $python_executable1 $full_path --env_name "LunarLander-v2" --visualize False --n_steps 100000 --frame_interval 10000 --n_environments 1 --algorithm "DQN" --gamma 0.95 --hidden_size 32 --lr 0.001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --batch_size 1 --use_replay False


# Check the exit code of the Python script
if ($LASTEXITCODE -eq 0) {
    Write-Output "$test_name script completed successfully."
} else {
    Write-Output "$test_name script encountered an error with exit code $LASTEXITCODE."
}