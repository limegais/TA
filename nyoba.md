#!/usr/bin/env python3
"""
Particle Swarm Optimization untuk Smart Light
"""

import random
from datetime import datetime

class PSOOptimizer:
    def __init__(self):
        # PSO Parameters
        self.n_particles = 20
        self.n_iterations = 30
        self.w = 0.7      # Inertia
        self.c1 = 1.5     # Cognitive
        self.c2 = 1.5     # Social
        
        # PWM Range
        self.pwm_min = 0
        self.pwm_max = 255
        
        # Target lux
        self.target_lux = {
            'reading': 500,
            'working': 400,
            'relaxing': 200,
            'sleeping': 50,
            'away': 0
        }
        
        # Weights
        self.comfort_weight = 0.5
        self.energy_weight = 0.3
        self.stability_weight = 0.2
        
        self.history = []
        self.last_pwm = 0
    
    def fitness(self, pwm, current_lux, target_lux, occupancy):
        """Hitung fitness (Higher = Better)"""
        
        # Estimated lux contribution
        estimated_contribution = pwm * 2
        estimated_total = current_lux + estimated_contribution
        
        # === COMFORT ===
        if not occupancy:
            comfort = 100 if pwm < 20 else (100 - pwm / 2.55)
        else:
            lux_diff = abs(estimated_total - target_lux)
            comfort = max(0, 100 - (lux_diff / 5))
        
        # === ENERGY ===
        energy = 100 - (pwm / 2.55)
        
        # === STABILITY ===
        change = abs(pwm - self.last_pwm)
        stability = max(0, 100 - change)
        
        # === FINAL ===
        score = (
            self.comfort_weight * comfort +
            self.energy_weight * energy +
            self.stability_weight * stability
        )
        
        return score
    
    def optimize(self, current_lux, occupancy, activity='working'):
        """Jalankan PSO"""
        target = self.target_lux.get(activity, 300)
        if not occupancy:
            target = 0
        
        # Initialize
        particles = [random.uniform(self.pwm_min, self.pwm_max) 
                    for _ in range(self.n_particles)]
        velocities = [random.uniform(-50, 50) 
                     for _ in range(self.n_particles)]
        
        p_best = particles.copy()
        p_best_fitness = [self.fitness(p, current_lux, target, occupancy) 
                         for p in particles]
        
        g_best_idx = p_best_fitness.index(max(p_best_fitness))
        g_best = p_best[g_best_idx]
        g_best_fitness = p_best_fitness[g_best_idx]
        
        # PSO iterations
        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (p_best[i] - particles[i]) +
                    self.c2 * r2 * (g_best - particles[i])
                )
                
                particles[i] += velocities[i]
                particles[i] = max(self.pwm_min, min(self.pwm_max, particles[i]))
                
                fit = self.fitness(particles[i], current_lux, target, occupancy)
                
                if fit > p_best_fitness[i]:
                    p_best[i] = particles[i]
                    p_best_fitness[i] = fit
                
                if fit > g_best_fitness:
                    g_best = particles[i]
                    g_best_fitness = fit
        
        optimal_pwm = int(round(g_best))
        self.last_pwm = optimal_pwm
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'input': {'lux': current_lux, 'occupancy': occupancy, 'activity': activity},
            'output': optimal_pwm,
            'fitness': g_best_fitness
        })
        
        return optimal_pwm, g_best_fitness
    
    def get_recommendation(self, current_lux, occupancy, activity='working'):
        """Dapatkan rekomendasi PWM"""
        optimal_pwm, fitness = self.optimize(current_lux, occupancy, activity)
        
        reasons = []
        if not occupancy:
            reasons.append("Tidak ada orang")
        if current_lux < 100:
            reasons.append(f"Gelap ({current_lux:.0f} lux)")
        elif current_lux > 500:
            reasons.append(f"Terang ({current_lux:.0f} lux)")
        reasons.append(f"Activity: {activity}")
        
        return {
            'recommended_pwm': optimal_pwm,
            'recommended_percent': int(optimal_pwm * 100 / 255),
            'target_lux': self.target_lux.get(activity, 300),
            'confidence': fitness / 100,
            'reason': "; ".join(reasons)
        }
