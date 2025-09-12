import xml.etree.ElementTree as ET
from collections import defaultdict

tripinfo_path = 'C:/FTO-Sim/Additionals/OJ_T-ITS/out/tripinfo.xml'

# Parse XML file
tree = ET.parse(tripinfo_path)
root = tree.getroot()

# Dictionary to store waiting times for each flow
flow_wait_times = defaultdict(list)

# Iterate through all tripinfo elements
for tripinfo in root.findall('tripinfo'):
    # Extract vehicle ID and waiting time
    vehicle_id = tripinfo.get('id')
    waiting_time = float(tripinfo.get('waitingTime', 0))
    
    # Extract flow ID from vehicle ID (e.g., "flow_0.0" -> "flow_0")
    flow_id = vehicle_id.rsplit('.', 1)[0]
    
    # Store waiting time for this flow
    flow_wait_times[flow_id].append(waiting_time)

# Calculate and print average waiting times per flow
print("Average waiting times per flow:")
for flow_id, wait_times in flow_wait_times.items():
    avg_wait_time = sum(wait_times) / len(wait_times)
    print(f"{flow_id}: {avg_wait_time:.2f} seconds")


