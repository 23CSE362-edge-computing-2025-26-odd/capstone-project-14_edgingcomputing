import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_vibration_data(samples=25000, fault_percent=0.2, random_seed=42):
    """
    Simulate data based on real-world data taken from https://data.nasa.gov/dataset/ims-bearings
    """
    np.random.seed(random_seed)  # so we get same results each time
    
    # make normal vibration readings first
    normal_data = np.random.normal(0, 0.1, samples)  # mean=0, std=0.1
    
    # faulty count
    num_faults = int(fault_percent * samples)
    
    # faulty positions
    fault_positions = np.random.choice(samples, num_faults, replace=False)
    
    # really bad faults - 0.3%
    really_bad_faults = int(0.03 * num_faults)
    if really_bad_faults < 1:
        really_bad_faults = 1 # at least one really bad fault
    
    bad_fault_positions = np.random.choice(fault_positions, really_bad_faults, replace=False)
    mild_fault_positions = [] # mild faults
    for pos in fault_positions:
        if pos not in bad_fault_positions:
            mild_fault_positions.append(pos)
    
    # copying data to not interfere with original when modifying it
    vibration_data = normal_data.copy()
    
    # mild faults
    for i in mild_fault_positions:
        if np.random.random() < 0.5:
            vibration_data[i] = np.random.uniform(-0.6, -0.4)  # negative fault
        else:
            vibration_data[i] = np.random.uniform(0.4, 0.6)   # positive fault
    
    # really bad faults
    for i in bad_fault_positions:
        if np.random.random() < 0.5:
            vibration_data[i] = np.random.uniform(-1.0, -0.8)  # really negative
        else:
            vibration_data[i] = np.random.uniform(0.8, 1.0)    # really positive
    
    return vibration_data, fault_positions

# generate the data
vibration_values, fault_indices = make_vibration_data(25000, 0.2, 42)

# save as csv file
df = pd.DataFrame({'vibration': vibration_values})
df.to_csv('Dataset/Data/simulated_vibration.csv', index=False)
print(f"Saved data points to simulated_vibration.csv")

# visualizing the data
plt.figure(figsize=(12, 8))

# histogram
plt.subplot(2, 2, 1)
plt.hist(vibration_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Vibration Values')
plt.xlabel('Vibration Amplitude')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# time vs fault occurence
plt.subplot(2, 2, 2)
sample_data = vibration_values[:3000]
plt.plot(sample_data, linewidth=0.8, color='green')
plt.title('Vibration Over Time (First 1000 readings)')
plt.xlabel('Time Index')
plt.ylabel('Vibration Amplitude')
plt.grid(True, alpha=0.3)

# scatter plot
plt.subplot(2, 2, 3)
normal_indices = []
for i in range(len(vibration_values)):
    if i not in fault_indices:
        normal_indices.append(i)

point_count = 10000
sample_normal = np.random.choice(normal_indices, min(int(point_count*0.8), len(normal_indices)), replace=False)
sample_faulty = np.random.choice(fault_indices, min(int(point_count*0.2), len(fault_indices)), replace=False)

plt.scatter(sample_normal, vibration_values[sample_normal], 
           c='blue', alpha=0.6, s=2, label='Normal')
plt.scatter(sample_faulty, vibration_values[sample_faulty], 
           c='red', alpha=0.8, s=4, label='Faulty')
plt.title('Normal vs Faulty Readings')
plt.xlabel('Sample Index')
plt.ylabel('Vibration Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()