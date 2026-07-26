import gymnasium as gym
import csv

from environment.modified_lunarlander import ModifiedLunarLander


EPISODES = 200


env = ModifiedLunarLander(
    gym.make("LunarLander-v3")
)


for episode in range(EPISODES):

    observation, info = env.reset(seed=episode)

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)


stats = env.get_statistics()
csv_filename = "verification_logs.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file) 
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Episodes", EPISODES])
    writer.writerow(["Thruster Actions", stats["Thruster Actions"]])
    writer.writerow(["Failed Thrusters", stats["Failed Thrusters"]])
    writer.writerow(["Failure Rate (%)", round(stats["Failure Rate"] * 100, 2)])
    writer.writerow(["Fuel Penalties", stats["Fuel Penalties"]])
    writer.writerow(["Safe Landings", stats["Safe Landings"]])
    writer.writerow(["Failure Rate Approx. 15%",abs(stats["Failure Rate"] - 0.15) < 0.02,])
    writer.writerow(["Fuel Penalties == Thruster Actions",stats["Fuel Penalties"] == stats["Thruster Actions"],])

print(f"\nCSV log saved as '{csv_filename}'")

print("\nVerification Statistics")
print("=" * 40)

print(f"Total Thruster Actions : {stats['Thruster Actions']}")
print(f"Failed Thrusters       : {stats['Failed Thrusters']}")
print(f"Failure Rate           : {stats['Failure Rate']*100:.2f}%")
print(f"Fuel Penalties Applied : {stats['Fuel Penalties']}")
print(f"Safe Landings          : {stats['Safe Landings']}")

print("\nExpected:")
print("Failure Rate ≈ 15%", abs(stats["Failure Rate"] - 0.15) < 0.02)
print("Fuel Penalties == Thruster Actions", stats["Fuel Penalties"] == stats["Thruster Actions"])

env.reset()