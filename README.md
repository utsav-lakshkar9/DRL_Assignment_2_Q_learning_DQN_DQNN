# Robust Reinforcement Learning under Stochastic Action Failure

## Question

The LunarLander-v3 environment in Gymnasium consists of:

### State Space
- 8-dimensional continuous state vector representing position, velocity, orientation, angular velocity, and leg-contact information.

### Action Space

| Action | Description |
|---:|---|
|0|Do nothing|
|1|Fire left orientation engine|
|2|Fire main engine|
|3|Fire right orientation engine|

Create a custom `gym.Wrapper` around `LunarLander-v3`.

## Environment Modifications

### Step 1: Receive the Agent's Action
Store the selected action before modification.

### Step 2: Simulate Intermittent Engine Failure
- If `a=0`, execute unchanged.
- If `a∈{1,2,3}`:
  1. Sample `r` uniformly from `[0,1)`.
  2. If `r<0.15`, execute action `0`.
  3. Otherwise execute the original action.

The agent must not know whether replacement occurred.

### Step 3
Execute the action and obtain:
- observation
- base reward `R_base`
- terminated
- truncated
- info

### Step 4: Modified Reward

```text
R = R_base - 0.3 * I(a ∈ {1,2,3}) + B
```

where `B` is

```text
50 if safe landing occurs
0 otherwise
```

Fuel penalty depends on the **selected** action, not the executed action.

### Step 5: Safe Landing Bonus

Award +50 only if:

- `terminated == True`
- `truncated == False`
- `observation[6] == 1`
- `observation[7] == 1`
- `abs(observation[2]) < 0.10`
- `abs(observation[3]) < 0.10`
- `abs(observation[4]) < 0.10`

Otherwise `B=0`.

### Step 6
Return:
- observation
- modified reward
- terminated
- truncated
- info

No extra information should be added to `info`.

## Tasks

### (a) Modified Environment Implementation and Verification (2.5 Marks)

Verify:
1. ~15% thruster actions become Do Nothing.
2. Fuel penalty always applies to attempted thruster actions.
3. Safe landing bonus is awarded only when all criteria are satisfied.

### (b) DQN (4 Marks)

Train DQN on:
1. Modified environment
2. Original environment

using identical hyperparameters, architecture, optimizer, seed and training duration.

### (c) DDQN (4 Marks)

Implement DDQN with identical setup. Only target computation differs.

Train on both environments.

### (d) Performance Evaluation (2 Marks)

Compare:
- DQN – Original
- DDQN – Original
- DQN – Modified
- DDQN – Modified

Plot:
1. Episode reward
2. Average predicted Q-value
3. Successful landing rate (100-episode moving average)
4. Average thruster activations

Interpret each plot.

### (e) Discussion (2.5 Marks)

Answer:

1. Does engine failure increase DQN/DDQN Q-value differences?
2. Why does stochastic action failure make credit assignment harder?
3. Does fuel penalty encourage conservative landings?
4. Which algorithm performs better under failures?
5. One limitation and one improvement.
