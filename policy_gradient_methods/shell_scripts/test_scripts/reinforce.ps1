$test_name = "Reinforce plain"
echo "starting $test_name"
$python_executable1 = "C:\Users\Mads-\miniconda3\envs\RL_special_course\python.exe"

& $python_executable1 C:\Users\Mads-\Documents\universitet\4_sem\RL\RL_special_course\policy_gradient_methods\main.py --seed 3 --n_episodes 2 --n_environments 2 --learning_algorithm "REINFORCE" --visualize False

# Check the exit code of the Python script
if ($LASTEXITCODE -eq 0) {
    Write-Output "$test_name script completed successfully."
} else {
    Write-Output "$test_name script encountered an error with exit code $LASTEXITCODE."
}