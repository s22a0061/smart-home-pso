import numpy as np
import random
from src.metrics import calculate_metrics

class Particle:
    def __init__(self, num_tasks):
        # Position: random start hour (0-23) for each shiftable task
        self.position = np.array([random.uniform(0, 23) for _ in range(num_tasks)])
        self.velocity = np.zeros(num_tasks)
        self.best_pos = np.copy(self.position)
        self.best_fitness = float('inf')
        self.fitness = float('inf')

class PSOOptimizer:
    def __init__(self, fixed_tasks, shiftable_tasks, swarm_size=30, w=0.5, c1=1.5, c2=1.5):
        self.fixed = fixed_tasks
        self.shiftable = shiftable_tasks
        self.w = w   # Inertia
        self.c1 = c1 # Cognitive constant
        self.c2 = c2 # Social constant
        self.swarm = [Particle(len(shiftable_tasks)) for _ in range(swarm_size)]
        self.gbest_pos = None
        self.gbest_fitness = float('inf')

    def optimize(self, iterations=50):
        history = []
        for _ in range(iterations):
            for particle in self.swarm:
                # Calculate Fitness (Cost + Penalty for 5kW limit)
                # Note: calculate_metrics should be updated to return (fitness, cost, discomfort, load)
                fit_val, _, _, _ = calculate_metrics(particle.position, self.fixed, self.shiftable)
                
                # Update Personal Best
                if fit_val < particle.best_fitness:
                    particle.best_fitness = fit_val
                    particle.best_pos = np.copy(particle.position)
                
                # Update Global Best
                if fit_val < self.gbest_fitness:
                    self.gbest_fitness = fit_val
                    self.gbest_pos = np.copy(particle.position)
            
            # Move Particles
            for particle in self.swarm:
                r1, r2 = random.random(), random.random()
                particle.velocity = (self.w * particle.velocity + 
                                   self.c1 * r1 * (particle.best_pos - particle.position) +
                                   self.c2 * r2 * (self.gbest_pos - particle.position))
                particle.position = np.clip(particle.position + particle.velocity, 0, 23)
            
            history.append(self.gbest_fitness)
        return self.gbest_pos, history
