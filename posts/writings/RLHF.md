# So Many RLHF Algorithms

Author: Webster Bei Yijie

## The Basic Setup of RL
In RL, you have some agent interacting with the environment. The agent observes the environment and issues an action on the environment, whereas the environment gives the reward and transition to a new state when acted upon. The goal of RL is to get a smart agent that maximizes the reward it receives.  
Agent issues actions based on a Policy, which is typically represented by a model $\pi{(a|s;\theta)}$, meaning that when given an observation of the environment $s$ as input, the policy will dictate a probability for taking each action across all the possible actions to take (then you can do anything you like with this probability map, e.g. take max prob action, or sample across probs etc). The Policy is parameterized by $\theta$ (also the learnable parameters). 

## Policy Gradient
Given the setup above, we want to find good $\theta$ so that we receive more rewards from the environment. However, we cannot just sum up all the reward values and do gradient descent to optimize for $\theta$ because rewards come out of the environment and loss function computed over rewards will not be differentiable since the environment is not differentiable.  
To make the learning possible, Policy Gradient is introduced. I will only present the conclusion here, and show you in code (which might be more useful) of how this can be implemented.

### Policy Gradient Theorem
$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}} [\nabla \ln{\pi{(a|s;\theta)}} Q_{\pi}(s,a)]
$$

In the above equation, $\mathcal{J}(\theta)$ is some form of the total reward we want to maximize, $Q_{\pi}(s,a)$ is some measure of "How good it is to be taking some action $a$ given observation of the environment $s$". In actual implementation of RL algorithms, $Q_{\pi}(s,a)$ is something we want to estimate in various smart ways, whereas $\nabla \ln{\pi{(a|s;\theta)}}$ can be directly calculated with autograd.  
In ML, expectations $\mathbb{E}[f(x)]$ are always estimated with samples. That means you can calculate the values of $f(x)$ for multiple values of $x$ and average/sum them to get $\mathbb{E}[f(x)]$. We do the same in actual implementation with minibatch.

### Policy Gradient Theorem Intuition
The intuition behind Policy Gradient Theorem is straightforward: move the parameter $\theta$ in a direction that increases/decreases the probability of outputting action $a$ proportional to "How good it is to output action $a$ when given observation of the environment $s$". So if $Q_{\pi}(s,a)$ is positive, we increase the probability of outputting $a$; if $Q_{\pi}(s,a)$ is negative, we decrease the probability. If $Q_{\pi}(s,a)$ very positive, we increase the probability of outputting $a$ by a large fraction, and otherwise increase it only a little.

### Policy Update
Imagine that you collected N samples somehow, represented by three tensors of shape [N,]:  
```Python
state_tensor = [s1, s2, s3, ..., sn] # [N, D] float tensor, some representation of the observation on the environment
actions_took = [a1, a2, a3,...,an] # int tensor
quality_of_action = [q1, q2, q3,...,qn] # float tensor
```
Don't worry about how you got those samples yet, we will talk about that later.  
Also imagine that you defined a neural network as the policy model, and it takes in some input $s$ and outputs a logits tensor of shape [N, A] where N is batch size and A is the total number of possible actions to take.  

```Python
logits = policy_model(s) # Output shape [N,A]
probs = torch.softmax(logits, dim=1) # Convert logits to probability across the action space
log_probs = probs.log() # log probs
log_probs_on_chosen_actions = torch.gather(log_probs, dim=1, actions_took.unsqueeze(1)) # Take the log prob value of the chosen actions for all examples
loss = - log_probs_on_chosen_actions * quality_of_action.unsqueeze(1) # logprobs * Q
loss.sum().backward()
optimizer.step()
```
So it is literally using "log probs of chosen action $a$" as part of the loss function, and scale those by reward. Notice that throughout the whole process, your `quality_of_action` (or Q in the equation) are essentially constants that you somehow got from interacting with the environment, and can be treated as constants from the perspective of gradient descent.

### Rollouts
To complete the picture, we need the last piece: how do we get the `actions_took` and `quality_of_action` tensors. The very short answer is: through simulation. So we simulate an agent (based on the Policy) and interact with the environment to get some rewards. This process is called `rollout`.  
The idea is that: given that you already have a Policy model (though it could be crappy in the beginning or even random), you can follow the probability distribution dictated by the Policy model to generate actions, act it upon the environment, collect some reward and repeat until you can no longer perform actions. This would give you a sequence of:  
$$
s_1, a_1, r_2, s_2, a_2, r_3, s_3, ..., r_t, s_t
$$
where $s_i$ represents the state (observation of the environment), $a_i$ represents the action that the agent took and $r_i$ represents the reward that the agent received at each step. The sequence above is called the `trajectory`.  
Now, given the trajectory, you can flatten it to give you the training data:
```Python
state_tensor = [s1, s2, s3, ...] # [N, D] float tensor, some representation of the observation on the environment
actions_took = [a1, a2, a3,...] # int tensor
quality_of_action = [q1, q2, q3,...] = [f(r2,r3...), f(r3,r4...), f(r4,r5...),...] # float tensor
```
`state_tensor` and `actions_took` are more straightforward, but what is quality_of_action? Well, it is pretty arbitrary IMO. Intuitively, the chosen action is good if:
1. It gives me a good immediate reward, and/or
2. It leads me to a state where I can potentially get higher total rewards from the new state onwards

A common way to encode the above two intuition is: 
$$
q_i = r_{i+1} + \gamma r_{i+2} + \gamma^2 r_{r+3} + ...
$$
This is called the discounted reward, where $\gamma$ is a number between 0 and 1. 

### Putting Everything Together
Repeat the steps of rollouts and policy update, magic will happen that a randomly initialized Policy model will converge and get better and better.

## The Innovations of the Different RL Algorithms
What we described above is essentially REINFORCE algorithm. There are various other RL algorithms that you might have heard of (let's focus on the ones that tries to model the Policy directly): Actor-Critic, TRPO, PPO, GRPO, RLOO, REINFORCE++. 

All these methods propose something innovative and demonstrate superiority against at-the-time SOTA methods in their papers, and there are a lot of nuances. But in this article, let's skip all the details and just focus on this question: "how are they different from the vanilla Policy Gradient / REINFORCE".   
First, let's take a step back to look at the Policy Gradient again:  

$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}} [\nabla \ln{\pi{(a|s;\theta)}} Q_{\pi}(s,a)]
$$
It seems that we need two things:
1. Some gradient over the policy on taken action: $\nabla \ln{\pi{(a|s;\theta)}}$
2. Some measure of how good the chosen action is given the observed state $s$: $Q_{\pi}(s,a)$

From a pure engineering perspective (i.e. ignoring most of the mathematical rationales for doing A vs B), and focus on the actual differences reflected in implementations, innovations mostly come from modifying the two things above.  

### REINFORCE with Baseline
It was found that changing our definition of `quality_of_action` from just `how good this action is` to `how good this action is compared to a baseline` reduces the gradient estimate variance. We call this new `quality_of_action` advantage (advantage of taking this particular action vs the average case), usually denoted as:
$$
A(s,a) = Q(s,a) - V(s)
$$
where $V(s)$ is the baseline. One way of getting the baseline is just to use the average $Q(s,a)$ value for different values of $a$.
### Actor-Critic
Keeps the concept of `advantage`, and replaces $V(s)$ estimate itself with a model. Each time we update the Policy, we also update the $V(s)$ model to get better and better baseline estimate.

### Generalized Advantage Estimation
We get another upgrade to the advantage function estimate. In Actor-Critic, we have:
$$
\begin{aligned}
A(s_t,a_t)^{(1)} &= Q(s_t, a_t) - V(s_t) \\ 
&= r_t + \gamma V(s_{t+1}) - V(s_t)
\end{aligned}
$$
So we can get an estimate with a minimal rollout length of 2 (i.e. you stop at $s_{t+1}$). However, with longer trajectory, you can get:
$$
\begin{aligned}
A(s_t,a_t)^{(2)} &= Q(s_t, a_t) - V(s_t) \\
&= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)
\end{aligned}
$$
$$
\begin{aligned}
A(s_t,a_t)^{(3)} &= Q(s_t, a_t) - V(s_t) \\
&= r_t + \gamma r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+3}) - V(s_t)
\end{aligned}
$$
So on so forth.  
GAE is the exponentially weighted average of all the possible $A(s_t,a_t)$ controlled by another variable $\lambda$:
$$
GAE = (1-\lambda)(A(s_t,a_t)^{(1)} + \lambda A(s_t,a_t)^{(2)} + \lambda^2A(s_t,a_t)^{(3)} + ...)
$$
Ok, very complicated, but essentially it is trying to get a better estimate for the advantage.
### TRPO
The three variations we talked above modified the `Some measure of how good the chosen action is given the observed state $s$` part. TRPO is different that it primarily modifies the base gradient calculations.  
Again revisiting the Policy Gradient Theorem:
$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}} [\nabla \ln{\pi{(a|s;\theta)}} Q_{\pi}(s,a)]
$$
Notice that the expectation notation has the $\pi_{\theta}$ subscript, which means that we should be performing rollouts on the current Policy. However, we can actually only perform rollouts with the Policy model from the previous iteration so the estimate is biased. To correct for the bias, they take inspiration from `importance sampling` and changed the formula to:
$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} [\nabla \frac{\pi{(a|s;\theta_{new})}}{\pi{(a|s;\theta_{old})}} Q_{\pi}(s,a)]
$$
So sample/rollout from old Policy (Policy from the previous iteration) and use the factor $\frac{1}{\pi{(a|s;\theta_{old})}}$ to correct for the estimate. TRPO implementation uses the conjugate gradient algorithm.
Additionally, TRPO also puts constraint on how much the the update can go by constrainining on the KL divergence between old and new Policy.
### PPO
The idea of TRPO and PPO is similar, but PPO introduces a clipping mechanism to handle the Policy update constraint in TRPO and also gradient descent for optimization.  
The base gradient formula is the same:
$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} [\nabla \frac{\pi{(a|s;\theta_{new})}}{\pi{(a|s;\theta_{old})}} Q_{\pi}(s,a)]
$$
With clipping, the gradient becomes:
$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} [\nabla \min(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}A(s,a), \text{clip}(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon)A(s,a))]
$$
In actual implementation, $\pi{(a|s;\theta_{old})}$ is just a constant. The code would look like:
```Python
logprobs_diff = log_probs_on_chosen_actions - log_probs_on_same_chosen_actions_with_old_policy
ratio = torch.exp(logprobs_diff)
# Add clipping
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
losses = torch.max(-advantage * ratio, -advantage * clipped_ratio)
```
The key innovation in PPO is the clipping of the probability ratio to stay within $(1-\epsilon, 1+\epsilon)$, where $\epsilon$ is typically 0.2. This prevents the new policy from deviating too far from the old policy in a single update, providing more stable training compared to TRPO while being simpler to implement.

### RLOO
Remember that in PPO (or any Actor-Critic framework algo), you have a model for estimating $V(s)$ to be used as part of the GAE (or baseline). RLOO gets rid of the $V(s)$ model by performing multiple rollouts from the same state, and for each rollout, it uses the remaining rollouts' average reward as the baseline.  
To illustrate, imagine I'm at state $s$, and I pick different actions $a_1, a_2, a_3,..., a_k$ as my next action. These different action choices eventually led to $k$ different reward outcomes $r_1, r_2, r_3, ..., r_k$. Then
$$
baseline_i = \frac{\sum_{j=1}^{k} r_j \mathbb{I}[i\neq j]}{k-1} \\
advantage_i = r_i - baseline_i
$$
In Python code it looks something like:
```Python
rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
advantages = rlhf_reward - baseline
```
I just copied the above code from TRL RLOO trainer.

### GRPO
I haven't seen an implementation of it, but from the paper, it seems to be very similar to RLOO except that it uses the average of all responses as the baseline:
$$
baseline_i = \frac{\sum_{j=1}^{k} r_j}{k} \\
advantage_i = r_i - baseline_i
$$

### REINFORCE++
This is probably the latest one as of now. It introduced a couple tricks on top of PPO/RLOO:
1. Normalizes rewards
2. Normalizes advantages
3. Token Level KL (to be completely honest, I read the implementation from TRL for RLOO trainer and I don't know what it is doing. Interested readers can take a look. I guess I understand what the original author tries to do just that given kl_coef is a scalar and reward model only gives a single score per example not per token, so I do think token-level KL and sequence level KL are the same computationally.)

## References
1. https://lilianweng.github.io/posts/2018-02-19-rl-overview/ 
2. https://arxiv.org/pdf/1707.06347
3. https://arxiv.org/pdf/2501.03262v1
4. https://arxiv.org/pdf/2402.03300
5. https://arxiv.org/pdf/1502.05477
6. https://arxiv.org/pdf/2402.14740
7. https://github.com/huggingface/trl
8. https://github.com/OpenRLHF/OpenRLHF