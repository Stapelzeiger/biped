import xml.etree.ElementTree as ET

# Load your URDF file
urdf_file = "custom_robot_v3.urdf"  # Replace with your URDF file name
urdf_file = "robot_yaw_fixed_urdf_v2.urdf"
# Parse the URDF XML
tree = ET.parse(urdf_file)
root = tree.getroot()

# Initialize total mass
total_mass = 0.0

# Iterate over all <mass> elements
for mass in root.findall('.//mass'):
    value = float(mass.get('value'))
    total_mass += value

print(f"Total mass: {total_mass} kg")