$laptop_paths=False

$test_name = "Reinforce w. entropy"
echo "starting $test_name"



if ($laptop_paths) {
$python_executable1 = "C:\Users\Mads-\AppData\Local\pypoetry\Cache\virtualenvs\rl-NjavQPoj-py3.8\Scripts\python.exe"
$repo_path = "C:\Users\Mads-\Documents\Universitet\Kandidat\4_semester\RL"
} else {
$python_executable1 = "C:\Users\Mads-\miniconda3\envs\RL_special_course\python.exe"
$repo_path = "C:\Users\Mads-\Documents\universitet\4_sem\RL"
}

$file_path = "RL_special_course\policy_gradient_methods\main.py"
$full_path = Join-Path -Path $repo_path -ChildPath $file_path
echo $full_path

& $python_executable1 $full_path --seed 3 --n_episodes 2 --n_environments 2 --learning_algorithm "REINFORCE" --visualize False --entropy True --grad_clipping 1 --env_name "LunarLanderContinuous-v2"
& $python_executable1 $full_path --seed 3 --n_episodes 2 --n_environments 2 --learning_algorithm "REINFORCE" --visualize False --entropy True --grad_clipping 1 --env_name "CartPole-v1"

# Check the exit code of the Python script
if ($LASTEXITCODE -eq 0) {
    Write-Output "$test_name script completed successfully."
} else {
    Write-Output "$test_name script encountered an error with exit code $LASTEXITCODE."
}



