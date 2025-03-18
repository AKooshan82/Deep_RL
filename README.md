## Course homeworks and codes will be posted here during the 14032 semester
### HW 1
Implemented a basic environment and tried to familirize ourselves with the basics of RL
### HW 2
Using Tabular methods we implemented the Q-Learning and SARSA algorithm. We also exprimented with deep RL algorithms like DDQN and DQN and triend to understand them.
### HW 3
The goal of this assignment was to explore policy-based reinforcement learning methods, with a focus on policy gradient algorithms. In particular :
1. Compare policy search methods – Implement and compare REINFORCE with genetic algorithm-
based policy search to understand different optimization approaches.
2. Improve REINFORCE – Implement and analyze variants of REINFORCE, such as REINFORCE
with baseline, to see how they enhance learning stability and efficiency.
3. Apply REINFORCE to continuous action spaces – Modify the algorithm to work in environ-
ments with continuous actions, highlighting key differences from discrete action spaces.
4. Compare Policy Gradient (REINFORCE) vs. DeepQ-Network (DQN) – Evaluate the
strengths and weaknesses of policy gradient methods in contrast to DeepQ-Network.

An important note to keep in mind is the fact that when dealing with continouse action-space, the output of the ``QNetwork`` which is going to indicate which action you should take, given the present state, is not a single number anymore. It's parameters to a probability distribution from which you are going to chose and action. In most cases the probability distribution is a Gaussian with parameters $\mu$ and $\sigma$. So the outputs of your network is gonna be two numbers here. <br>
If you use a ``nn.Linear`` for both of these variables and try to compute them using one network, things might get messy because they are going to be correlated due to the linear nature of the neural network used in computing them. It will be like having one feature instead of two. <br>
To fix this issue it's better to always consider one of them as a parameter of the model by using ``nn.Parameter( ......)``.
### HW 4
We used the PPO, SAC, and DDPG algorithms for continuous action-space environments. For different notebooks and tasks we did the followings :
1. Implemented a PPO agent for a continuous action-space environment.
2. Implemented a SAC and a DDPG agent for a continuous action-space environment.
3. Analyzed and compare the performance of SAC,DDPG and PPO in a specific environment.<br>

One of the main challenges in this exercise was finding the right hyperparameters for the PPO agent and after so many trials I found the optimal ones by experimenting and using the [original paper](https://arxiv.org/abs/1707.06347). Setting the ``lr = 3e-4`` fixed the problem.<br>
Also confusing the critic loss function inputs was a problem. Instead of calculating the mean square error of ``state_value`` and ``return`` which is the correct thing to do based on the paper and common sense, I was wrongly computing the mse between ``state_value`` and ``rewards``.
