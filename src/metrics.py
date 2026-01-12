import numpy as np
import pandas as pd

# Malaysian TOU Tariff (TNB Example)
# Peak: 8:00 AM - 10:00 PM (0.35 RM)
# Off-Peak: 10:00 PM - 8:00 AM (0.20 RM)
TARIFF = [0.20]*8 + [0.35]*14 + [0.20]*2

def calculate_metrics(particle_position, df):
    """
    particle_position: list of start hours for shiftable appliances
    df: the project_benchmark_data dataframe
    """
    total_cost = 0
    total_discomfort = 0
    hourly_load = np.zeros(24)
    
    # 1. Process Fixed Appliances
    fixed = df[df['Is_Shiftable'] == False]
    for _, task in fixed.iterrows():
        start = int(task['Preferred_Start_Hour'])
        duration = int(task['Duration_Hours'])
        power = task['Avg_Power_kW']
        for h in range(start, start + duration):
            h_idx = h % 24
            hourly_load[h_idx] += power
            total_cost += power * TARIFF[h_idx]

    # 2. Process Shiftable Appliances (Based on PSO Particle Position)
    shiftable = df[df['Is_Shiftable'] == True]
    for i, (_, task) in enumerate(shiftable.iterrows()):
        start = int(round(particle_position[i])) % 24 # PSO gives the new start time
        duration = int(task['Duration_Hours'])
        power = task['Avg_Power_kW']
        preferred = task['Preferred_Start_Hour']
        
        # Calculate Discomfort
        total_discomfort += abs(start - preferred)
        
        # Calculate Cost and Load
        for h in range(start, start + duration):
            h_idx = h % 24
            hourly_load[h_idx] += power
            total_cost += power * TARIFF[h_idx]

    # 3. Handle Constraint: 5.0 kW Peak Power Limit
    penalty = 0
    if any(load > 5.0 for load in hourly_load):
        penalty = 1000  # High penalty for violating the 5.0kW limit
    
    return total_cost + penalty, total_discomfort, hourly_load
