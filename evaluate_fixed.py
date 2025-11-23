import subprocess
import os

print("\n--- Running Fixed-Time Signal Simulation ---")
print("Phase 1 (55s): South → North + West")
print("Phase 2 (55s): North → South + East")
print("Phase 3 (55s): West → East + North")
print("Phase 4 (30s): East → West + South")
print("\nTotal cycle: 204 seconds (3m 24s)\n")

# Set SUMO_HOME
os.environ['SUMO_HOME'] = '/usr/share/sumo'

# Run SUMO with GUI and fixed timing
cmd = [
    '/usr/share/sumo/bin/sumo-gui',
    '-n', 'single_intersection.net.xml',
    '-r', 'single_intersection.rou.xml',
    '--additional-files', 'fixed_timing.add.xml',
    '--start',  # Auto-start simulation
    '--quit-on-end',  # Close when simulation ends
]

subprocess.run(cmd, cwd='/home/wetcatto/.code/machine-learning-le')
print("\nSimulation finished.")
