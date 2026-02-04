
import random
from datetime import datetime

class GAOptimizer:
    def __init__(self):
        # GA Parameters
        self.population_size = 30
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # AC Parameters Range
        self.temp_min = 16
        self.temp_max = 30
        self.modes = ['cool', 'eco', 'fan', 'off']
        self.fan_speeds = ['low', 'medium', 'high', 'auto']
        
        # Weights for fitness function
        self.comfort_weight = 0.4
        self.energy_weight = 0.3
        self.preference_weight = 0.3
        
        # History
        self.history = []
    
    def create_individual(self):
        """Buat satu individu (solusi)"""
        return {
            'temp_setting': random.randint(self.temp_min, self.temp_max),
            'mode': random.choice(self.modes),
            'fan_speed': random.choice(self.fan_speeds)
        }
    
    def create_population(self):
        """Buat populasi awal"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual, current_temp, humidity, occupancy, time_of_day):
        """Hitung fitness score (Higher = Better)"""
        temp_setting = individual['temp_setting']
        mode = individual['mode']
        
        # === COMFORT SCORE ===
        if mode == 'off':
            comfort = 0 if current_temp > 28 else 50
        else:
            temp_diff = abs(current_temp - temp_setting)
            if 23 <= temp_setting <= 26:
                comfort = 100 - (temp_diff * 5)
            else:
                comfort = 80 - (temp_diff * 5)
        
        # Humidity factor
        if 40 <= humidity <= 60:
            comfort += 10
        
        comfort = max(0, min(100, comfort))
        
        # === ENERGY SCORE ===
        energy_map = {'off': 0, 'fan': 20, 'eco': 50, 'cool': 100}
        temp_factor = (temp_setting - self.temp_min) / (self.temp_max - self.temp_min)
        energy = 100 - energy_map.get(mode, 50) * (1 - temp_factor * 0.3)
        energy = max(0, min(100, energy))
        
        # === PREFERENCE SCORE ===
        preference = 50
        
        # Night time (22:00 - 06:00)
        if time_of_day >= 22 or time_of_day < 6:
            if mode in ['eco', 'off']:
                preference = 80
            elif mode == 'cool':
                preference = 40
        
        # Occupancy
        if not occupancy:
            if mode == 'off':
                preference = 100
            elif mode == 'eco':
                preference = 70
            else:
                preference = 30
        else:
            if mode == 'off' and current_temp > 27:
                preference = 20
            elif mode == 'cool' and current_temp < 25:
                preference = 40
        
        # === FINAL SCORE ===
        score = (
            self.comfort_weight * comfort +
            self.energy_weight * energy +
            self.preference_weight * preference
        )
        
        return score
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        parents = []
        
        for _ in range(2):
            candidates = random.sample(
                list(zip(population, fitness_scores)), 
                min(tournament_size, len(population))
            )
            winner = max(candidates, key=lambda x: x[1])
            parents.append(winner[0])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Crossover dua parent"""
        if random.random() < self.crossover_rate:
            return {
                'temp_setting': random.choice([parent1['temp_setting'], parent2['temp_setting']]),
                'mode': random.choice([parent1['mode'], parent2['mode']]),
                'fan_speed': random.choice([parent1['fan_speed'], parent2['fan_speed']])
            }
        return parent1.copy()
    
    def mutate(self, individual):
        """Mutasi individu"""
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            mutated['temp_setting'] = random.randint(self.temp_min, self.temp_max)
        
        if random.random() < self.mutation_rate:
            mutated['mode'] = random.choice(self.modes)
        
        if random.random() < self.mutation_rate:
            mutated['fan_speed'] = random.choice(self.fan_speeds)
        
        return mutated
    
    def optimize(self, current_temp, humidity, occupancy, time_of_day=None):
        """Jalankan GA optimization"""
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        population = self.create_population()
        best_ever = None
        best_fitness = -1
        
        for _ in range(self.generations):
            fitness_scores = [
                self.fitness(ind, current_temp, humidity, occupancy, time_of_day)
                for ind in population
            ]
            
            # Track best
            max_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_ever = population[max_idx].copy()
            
            # Create new population
            new_population = [best_ever.copy()] if best_ever else []
            
            while len(new_population) < self.population_size:
                parents = self.select_parents(population, fitness_scores)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Save history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'input': {'temp': current_temp, 'humidity': humidity, 'occupancy': occupancy},
            'output': best_ever,
            'fitness': best_fitness
        })
        
        return best_ever, best_fitness
    
    def get_recommendation(self, current_temp, humidity, occupancy):
        """Dapatkan rekomendasi setting AC"""
        best, fitness = self.optimize(current_temp, humidity, occupancy)
        
        if best is None:
            best = {'temp_setting': 24, 'mode': 'eco', 'fan_speed': 'auto'}
            fitness = 50
        
        reasons = []
        if not occupancy:
            reasons.append("Tidak ada orang")
        if current_temp > 28:
            reasons.append(f"Suhu tinggi ({current_temp}°C)")
        elif current_temp < 24:
            reasons.append(f"Suhu nyaman ({current_temp}°C)")
        if humidity > 70:
            reasons.append("Kelembaban tinggi")
        
        return {
            'recommended_temp': best['temp_setting'],
            'recommended_mode': best['mode'],
            'recommended_fan': best['fan_speed'],
            'confidence': fitness / 100,
            'reason': "; ".join(reasons) if reasons else "Kondisi normal"
        }
