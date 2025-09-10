import csv
import random

def generate_value():
    """Generate a value based on the specified probability distribution"""
    rand = random.random()  # Random number between 0 and 1
    
    if rand < 0.8:  # 80% probability
        return random.uniform(3, 5)
    elif rand < 0.97:  # 17% probability (0.8 + 0.17 = 0.97)
        return random.uniform(6, 10)
    else:  # 3% probability
        return random.uniform(15, 20)

def generate_csv(filename, num_rows=1000):
    """Generate CSV file with specified distribution"""
    data = []
    
    # Generate data
    for _ in range(num_rows):
        value = generate_value()
        data.append([round(value, 2)])  # Single column with rounded value
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['vibration'])  # Header row
        writer.writerows(data)
    
    return data

def verify_distribution(data):
    """Verify the distribution of generated values"""
    values = [row[0] for row in data]  # Extract values from single column
    
    range_3_5 = sum(1 for v in values if 3 <= v <= 5)
    range_6_10 = sum(1 for v in values if 6 <= v <= 10)
    range_15_20 = sum(1 for v in values if 15 <= v <= 20)
    total = len(values)
    
    print(f"Total values: {total}")
    print(f"3-5 range: {range_3_5} values ({range_3_5/total*100:.1f}%)")
    print(f"6-10 range: {range_6_10} values ({range_6_10/total*100:.1f}%)")
    print(f"15-20 range: {range_15_20} values ({range_15_20/total*100:.1f}%)")

# Generate the CSV file
filename = "./Dataset/Data/check_values.csv"
num_rows = 500  # Generate 10,000 rows for better distribution accuracy

print(f"Generating {num_rows} rows of data...")
data = generate_csv(filename, num_rows)
print(f"CSV file '{filename}' created successfully!")

# Verify the distribution
print("\nDistribution verification:")
verify_distribution(data)

# Show first 10 rows as preview
print(f"\nFirst 10 rows:")
for i, row in enumerate(data[:10]):
    print(f"Row {i+1}: {row[0]}")