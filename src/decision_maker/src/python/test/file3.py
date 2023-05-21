import csv

# Sample data
data = [
    [1, 'John', 25],
    [2, 'Emily', 32],
    [3, 'Michael', 18]
]

# CSV file path
csv_file = '/home/kenji/ws/explORB-SLAM-RL/src/decision_maker/src/python/test/data.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['ID', 'Name', 'Age'])

    # Write the data rows
    writer.writerows(data)

print(f"Data has been stored in {csv_file}")