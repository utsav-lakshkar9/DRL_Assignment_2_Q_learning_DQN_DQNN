import gymnasium as gym
import numpy as np

class ModifiedLunarLander(gym.Wrapper):
    """
    LunarLander wrapper implementing:
        • 15% stochastic engine failure
        • Fuel penalty
        • Safe landing bonus
    """

    def __init__(self, env):
        super().__init__(env)

        self.failure_probability = 0.15
        self.fuel_penalty = 0.3
        self.landing_bonus = 50

        # Verification statistics
        self.total_thruster_actions = 0
        self.failed_thruster_actions = 0
        self.total_penalties = 0
        self.total_safe_landings = 0

    def step(self, action):

        original_action = action

        # ----------------------------
        # Step 2: Simulate engine failure
        # ----------------------------
        executed_action = original_action

        if original_action in {1, 2, 3}:
            self.total_thruster_actions += 1

            if np.random.rand() < self.failure_probability:
                executed_action = 0
                self.failed_thruster_actions += 1

        # ----------------------------
        # Execute action
        # ----------------------------
        observation, base_reward, terminated, truncated, info = self.env.step(
            executed_action
        )

        reward = float(base_reward)

        # ----------------------------
        # Fuel penalty
        # ----------------------------
        if original_action in {1, 2, 3}:
            reward -= self.fuel_penalty
            self.total_penalties += 1

        # ----------------------------
        # Safe landing bonus
        # ----------------------------
        safe_landing = (
            terminated
            and not truncated
            and observation[6] == 1
            and observation[7] == 1
            and abs(observation[2]) < 0.10
            and abs(observation[3]) < 0.10
            and abs(observation[4]) < 0.10
        )

        if safe_landing:
            reward += self.landing_bonus
            self.total_safe_landings += 1

        return observation, reward, terminated, truncated, info

    def get_statistics(self):

        failure_rate = 0

        if self.total_thruster_actions > 0:
            failure_rate = (
                self.failed_thruster_actions
                / self.total_thruster_actions
            )

        return {
            "Thruster Actions": self.total_thruster_actions,
            "Failed Thrusters": self.failed_thruster_actions,
            "Failure Rate": failure_rate,
            "Fuel Penalties": self.total_penalties,
            "Safe Landings": self.total_safe_landings,
        }