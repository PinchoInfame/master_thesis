import numpy as np
import os

# Get the absolute path to the results file from the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "..", "results")
file_path = os.path.join(results_dir, "simulation_results.txt")
output_path = os.path.join(results_dir, "result_analysis.txt")

# Initialize data containers
centralized_data = {"Rg": [], "Robs": [], "Rcoll": [], "Time": []}
decentralized_data = {"Rg": [], "Robs": [], "Rcoll": [], "Time": []}

current_solver = None

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith("centralized"):
            current_solver = centralized_data
        elif line.startswith("decentralized"):
            current_solver = decentralized_data
        elif line.startswith("Rg:"):
            parts = line.split(", ")
            rg = float(parts[0].split(": ")[1])
            robs = float(parts[1].split(": ")[1])
            rcoll = float(parts[2].split(": ")[1])
            time = float(parts[3].split(": ")[1])
            current_solver["Rg"].append(rg)
            current_solver["Robs"].append(robs)
            current_solver["Rcoll"].append(rcoll)
            current_solver["Time"].append(time)

# Function to print and collect stats
def collect_stats(name, data):
    lines = [f"{name} Solver Statistics:"]
    for key in data:
        arr = np.array(data[key])
        mean = arr.mean()
        std = arr.std()
        lines.append(f"  {key}: mean = {mean:.2f}, std = {std:.2f}")
    return lines

# Collect and print results
centralized_stats = collect_stats("Centralized", centralized_data)
decentralized_stats = collect_stats("Decentralized", decentralized_data)

# Print to terminal
for line in centralized_stats + [""] + decentralized_stats:
    print(line)

# Save to file
with open(output_path, "a") as out_file:
    out_file.write("\n--- New Analysis ---\n")
    for line in centralized_stats + [""] + decentralized_stats:
        out_file.write(line + "\n")
