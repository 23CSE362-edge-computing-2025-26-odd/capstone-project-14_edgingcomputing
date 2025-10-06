import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# Machine counts in the same order as res*.txt files
counts = [400, 1000, 1500, 2000, 3000]

accuracies = []

# Automatically find res*.txt files and sort them by number
files = sorted(
    glob.glob("./fleet_anomaly_sim/analysis/res*.txt"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0][3:])
)

for file in files:
    with open(file, "r") as f:
        data = f.read().splitlines()
    
    original = np.array(eval(data[0]))
    predicted = np.array(eval(data[1]))

    accuracy = np.isin(predicted, original).sum() / len(predicted) * 100
    accuracies.append(accuracy)

    print(f"Fleet of {counts[len(accuracies)-1]} machines: {accuracy:.2f}%")

# Plot Machines vs Accuracy
plt.plot(counts, accuracies, marker='o')
plt.title("Machine Fleet Size vs Accuracy")
plt.xlabel("Number of Machines")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 100)  # force percentage range
plt.grid(True)
plt.show()
