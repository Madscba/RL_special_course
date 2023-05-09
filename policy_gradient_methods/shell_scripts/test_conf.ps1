#Wait for last job to complete
While (Get-Job -State "Running") {
    Log "Running test scripts" -v $info
    Start-Sleep 2
}

<# $python_executable1 = "C:\Users\Mads-\miniconda3\envs\RL_special_course\python.exe" #>
$python_executable1 = "C:\Users\Mads-\AppData\Local\pypoetry\Cache\virtualenvs\rl-NjavQPoj-py3.8\Scripts\python.exe"

$root_directory_1 = "policy_gradient_methods/shell_scripts/test_scripts"
# Define the scripts you want to run
$scripts = @(
    "AC.ps1",
    "AC_e.ps1",
    "reinforce.ps1",
    "reinforce_e.ps1"
)

# Start a job for each script
foreach ($script in $scripts) {
    $script_path = Join-Path $root_directory_1 $script
    Start-Job -FilePath $script_path
}

# Wait for all jobs to finish
Get-Job | Wait-Job

# Get the results of each job
Get-Job | Receive-Job

#clear jobs from job queue
Get-job | Remove-Job
