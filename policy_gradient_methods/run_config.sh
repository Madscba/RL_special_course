python policy_gradient_methods\main.py --seed 3 --n_episodes 400 --n_environments 10 --entropy True --learning_algorithm "AC"
python policy_gradient_methods\main.py --seed 3 --n_episodes 4000 --n_environments 10 --learning_algorithm "AC"

python policy_gradient_methods\main.py --seed 3 --n_episodes 4000 --n_environments 10 --entropy True --learning_algorithm "REINFORCE"
python policy_gradient_methods\main.py --seed 3 --n_episodes 4000 --n_environments 10 --learning_algorithm "REINFORCE"