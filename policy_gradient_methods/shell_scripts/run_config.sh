python policy_gradient_methods\main.py --seed 3 --n_episodes 10000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr


python policy_gradient_methods\main.py --seed 3 --n_episodes 1000 --n_environments 20 --learning_algorithm "REINFORCE" --grad_clipping 1 --entropy True


python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.001 --env_name CartPole-v1


python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.0001 --env_name LunarLanderContinuous-v2
python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.001 --env_name LunarLanderContinuous-v2

python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.0001 --env_name LunarLanderContinuous-v2 --hidden_size 64

python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.01
python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC"  --lr 0.0001


#Best performing AC so far:
python policy_gradient_methods\main.py --seed 3 --n_episodes 2000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.001

#RUN TOMORROW:

##decent performance
python policy_gradient_methods\main.py --seed 3 --n_episodes 5000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.001 --env_name LunarLanderContinuous-v2
python policy_gradient_methods\main.py --seed 3 --n_episodes 5000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.0001 --env_name LunarLanderContinuous-v2

python policy_gradient_methods\main.py --seed 3 --n_episodes 5000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.0001 --env_name LunarLanderContinuous-v2 --hidden_size 64

##pretty bad performance
python policy_gradient_methods\main.py --seed 3 --n_episodes 5000 --n_environments 1 --learning_algorithm "AC" --grad_clipping 1 --entropy True --lr 0.001 --env_name LunarLanderContinuous-v2 --hidden_size 64
