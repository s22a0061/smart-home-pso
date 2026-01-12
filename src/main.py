import pandas as pd
import numpy as np

df = pd.read_csv('project_benchmark_data.csv')

def calculate_total_power_profile(scheduled_start_times):
    # scheduled_start_times is the "Position" of a PSO particle
    # e.g., [14, 22] for two shiftable appliances
    
    hourly_power = np.zeros(24)
    
    # 1. Add Fixed Appliances (Cannot be moved)
    fixed = df[df['Is_Shiftable'] == False]
    for _, task in fixed.iterrows():
        start = int(task['Preferred_Start_Hour'])
        duration = int(task['Duration_Hours'])
        for h in range(start, start + duration):
            hourly_power[h % 24] += task['Avg_Power_kW']
            
    # 2. Add Shiftable Appliances (Times decided by PSO)
    shiftable = df[df['Is_Shiftable'] == True]
    for i, (_, task) in enumerate(shiftable.iterrows()):
        start = int(scheduled_start_times[i]) # Value from PSO particle
        duration = int(task['Duration_Hours'])
        for h in range(start, start + duration):
            hourly_power[h % 24] += task['Avg_Power_kW']
            
    return hourly_power
