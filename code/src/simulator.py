"""
Day 6-8: Fleet Selective Maintenance (FSM) Simulator
=====================================================
Simulates a fleet of machines with limited repair capacity.

Components:
    - Cost Model (Day 6): preventive, failure, waste costs
    - Greedy Scheduler (Day 7): prioritize by predicted RUL
    - Rolling Horizon Simulator (Day 8): day-by-day simulation

Game Rules:
    - N machines, each degrading independently
    - Maintenance capacity K per day (resource constraint)
    - Scheduler uses predicted RUL to decide who to maintain
    - Higher penalty for unexpected failure vs. preventive maintenance
"""

import numpy as np
import torch
import highspy
from src.data_generator import generate_single_curve


# ==================================================================
# Day 6: Cost Definitions
# ==================================================================

class CostConfig:
    """Fleet maintenance cost parameters."""
    
    def __init__(
        self,
        c_preventive=10.0,    # Cost of preventive maintenance
        c_failure=100.0,      # Cost of unexpected failure
        c_waste=1.0,          # Cost per unit of wasted remaining life
        capacity=2,           # Max machines maintained per day
        n_machines=20,        # Number of machines in fleet
        safety_threshold=15,  # RUL threshold below which to schedule maintenance
    ):
        self.c_preventive = c_preventive
        self.c_failure = c_failure
        self.c_waste = c_waste
        self.capacity = capacity
        self.n_machines = n_machines
        self.safety_threshold = safety_threshold
    
    def __repr__(self):
        return (
            f"CostConfig(c_prev={self.c_preventive}, c_fail={self.c_failure}, "
            f"c_waste={self.c_waste}, K={self.capacity}, N={self.n_machines}, "
            f"threshold={self.safety_threshold})"
        )


def calculate_cost(action, true_rul, config):
    """
    Calculate cost for a single machine maintenance decision.
    
    Args:
        action: 'maintain', 'failure', or 'none'
        true_rul: actual remaining useful life at decision time
        config: CostConfig instance
    
    Returns:
        cost: float
    """
    if action == 'failure':
        return config.c_failure
    elif action == 'maintain':
        # preventive_cost + waste_cost (wasting remaining life)
        return config.c_preventive + config.c_waste * max(0, true_rul)
    else:
        return 0.0


# ==================================================================
# Day 7: Greedy Scheduler
# ==================================================================

def greedy_scheduler(predicted_ruls, capacity, safety_threshold):
    """
    Greedy scheduling: maintain the most urgent machines first.
    (Legacy — kept for reference; superseded by highs_scheduler)
    
    Algorithm:
        1. Sort machines by predicted RUL (ascending — most urgent first)
        2. Select top `capacity` machines with RUL below safety threshold
        3. Return binary decision vector
    
    Args:
        predicted_ruls: np.array [n_machines], predicted RUL for each machine
        capacity:       int, max number of machines to maintain per day
        safety_threshold: float, RUL below which maintenance is triggered
    
    Returns:
        decisions: np.array [n_machines], 1 = maintain, 0 = no action
    """
    n = len(predicted_ruls)
    decisions = np.zeros(n, dtype=int)
    
    # Sort by predicted RUL (ascending — most urgent first)
    sorted_indices = np.argsort(predicted_ruls)
    
    maintained = 0
    for idx in sorted_indices:
        if maintained >= capacity:
            break
        if predicted_ruls[idx] < safety_threshold:
            decisions[idx] = 1
            maintained += 1
    
    return decisions


def highs_scheduler(predicted_ruls, capacity, safety_threshold, config):
    """
    Optimal maintenance scheduling via HiGHS MILP solver.

    Formulation (binary integer program):
        min  Σ_i  x_i · (m_i − f_i)
        s.t. Σ_i  x_i  ≤  capacity
             x_i ∈ {0, 1}   for all i

    where:
        m_i = c_preventive + c_waste · pred_rul_i   (cost if we maintain)
        f_i = c_failure · max(0, 1 − pred_rul_i / safety_threshold)
              (expected failure risk cost if we do NOT maintain)

    A negative coefficient (m_i − f_i < 0) means maintenance is cheaper
    than the expected failure cost, so the solver will try to set x_i = 1.

    Args:
        predicted_ruls:   np.array [n_machines], predicted RUL for each machine
        capacity:         int, max number of machines to maintain per day
        safety_threshold: float, RUL below which failure risk is nonzero
        config:           CostConfig instance with cost parameters

    Returns:
        decisions: np.array [n_machines], 1 = maintain, 0 = no action
    """
    n = len(predicted_ruls)
    decisions = np.zeros(n, dtype=int)

    # ---- Build HiGHS model ----
    h = highspy.Highs()
    h.silent()  # suppress solver output

    # Add n binary decision variables x_0 … x_{n-1}
    inf = highspy.kHighsInf
    lower = np.zeros(n)
    upper = np.ones(n)

    # Compute objective coefficients
    obj = np.empty(n)
    for i in range(n):
        m_cost = config.c_preventive + config.c_waste * predicted_ruls[i]
        f_risk = config.c_failure * max(0.0, 1.0 - predicted_ruls[i] / safety_threshold)
        obj[i] = m_cost - f_risk  # negative ⇒ should maintain

    h.addVars(n, lower, upper)

    # Set objective (minimize)
    cost_array = obj.tolist()
    for i in range(n):
        h.changeColCost(i, cost_array[i])

    # Mark all variables as integer (binary)
    integrality = [highspy.HighsVarType.kInteger] * n
    for i in range(n):
        h.changeColIntegrality(i, integrality[i])

    # Capacity constraint: Σ x_i ≤ capacity
    row_indices = np.arange(n, dtype=np.int32)
    row_values = np.ones(n)
    h.addRow(-inf, float(capacity), n, row_indices, row_values)

    # ---- Solve ----
    h.run()

    status = h.getInfoValue("primal_solution_status")[1]
    # 2 = feasible
    if status == 2:
        sol = h.getSolution()
        for i in range(n):
            if sol.col_value[i] > 0.5:
                decisions[i] = 1
    else:
        # Fallback to greedy if solver fails
        decisions = greedy_scheduler(predicted_ruls, capacity, safety_threshold)

    return decisions


# ==================================================================
# Day 8: Rolling Horizon Simulator
# ==================================================================

class Machine:
    """Represents a single machine in the fleet."""
    
    def __init__(self, machine_id, seed=None):
        self.machine_id = machine_id
        self.reset(seed)
    
    def reset(self, seed=None):
        """Initialize or reset machine with a fresh degradation curve."""
        rng = np.random.RandomState(seed)
        deg_rate = rng.uniform(0.3, 0.7)
        noise_base = rng.uniform(0.2, 0.5)
        noise_growth = rng.uniform(0.003, 0.008)
        
        health, rul, ft = generate_single_curve(
            max_life=300,
            degradation_rate=deg_rate,
            noise_base=noise_base,
            noise_growth=noise_growth,
            initial_health=100.0,
            failure_threshold=0.0,
            seed=seed
        )
        
        self.health_full = health
        self.rul_full = rul
        self.failure_time = ft
        self.current_step = 0
        self.status = 'running'  # 'running', 'failed', 'maintained'
    
    @property
    def current_health(self):
        if self.current_step < len(self.health_full):
            return self.health_full[self.current_step]
        return 0.0
    
    @property
    def true_rul(self):
        if self.current_step < len(self.rul_full):
            return self.rul_full[self.current_step]
        return 0.0
    
    def get_health_window(self, window_size=30):
        """Get the last `window_size` health readings, padded if needed."""
        end = self.current_step + 1
        start = max(0, end - window_size)
        
        window = self.health_full[start:end]
        
        # Pad with initial health if not enough readings
        if len(window) < window_size:
            pad = np.full(window_size - len(window), self.health_full[0])
            window = np.concatenate([pad, window])
        
        return window
    
    def age_one_day(self):
        """Advance one time step. Returns True if failed."""
        if self.status != 'running':
            return False
        
        self.current_step += 1
        
        if self.current_step >= len(self.health_full) or self.current_health <= 0:
            self.status = 'failed'
            return True
        
        return False


class FleetSimulator:
    """
    Rolling horizon fleet maintenance simulator.
    
    Each day:
        1. Machines age (advance one step)
        2. Check for failures
        3. Model predicts RUL for running machines
        4. Scheduler decides who to maintain
        5. Settle costs (failure penalty + maintenance cost + waste)
        6. Reset maintained/failed machines (get repaired/replaced)
    """
    
    def __init__(self, config, model=None, rul_max=1.0,
                 window_size=30, device='cpu', seed=42):
        """
        Args:
            config:      CostConfig instance
            model:       trained BiLSTM_RUL model (or None for oracle)
            rul_max:     RUL normalization factor
            window_size: input window size for the model
            device:      'cpu' or 'cuda'
            seed:        random seed
        """
        self.config = config
        self.model = model
        self.rul_max = rul_max
        self.window_size = window_size
        self.device = device
        self.seed = seed
        
        # Initialize fleet
        self.machines = []
        for i in range(config.n_machines):
            m = Machine(machine_id=i, seed=seed + i * 1000)
            self.machines.append(m)
        
        # Tracking
        self.daily_costs = []
        self.total_failures = 0
        self.total_maintenances = 0
        self.total_cost = 0.0
        self.day_count = 0
        self.log = []
    
    def predict_rul(self, machine):
        """
        Predict RUL for a single machine using the model.
        If no model, use true RUL (oracle baseline).
        """
        if self.model is None:
            # Oracle: use true RUL
            return machine.true_rul
        
        # Get health window and normalize
        window = machine.get_health_window(self.window_size)
        window_norm = window / 100.0
        
        # Convert to tensor: [1, window_size, 1]
        x = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).item()
        
        # Denormalize
        pred_rul = pred_norm * self.rul_max
        return max(pred_rul, 0.0)
    
    def run(self, n_days=1000, verbose=False):
        """
        Run the simulation for n_days.
        
        Returns:
            results dict with total_cost, total_failures, etc.
        """
        reset_seed_counter = self.seed + 10000
        
        for day in range(n_days):
            self.day_count += 1
            day_cost = 0.0
            day_failures = 0
            day_maintenances = 0
            
            # Step 1: Age all running machines
            for m in self.machines:
                if m.status == 'running':
                    failed = m.age_one_day()
                    if failed:
                        # Failure penalty
                        cost = calculate_cost('failure', 0, self.config)
                        day_cost += cost
                        day_failures += 1
            
            # Step 2: Predict RUL for running machines
            running_indices = [i for i, m in enumerate(self.machines)
                             if m.status == 'running']
            
            if running_indices:
                predicted_ruls = np.zeros(len(self.machines))
                for i in running_indices:
                    predicted_ruls[i] = self.predict_rul(self.machines[i])
                
                # Set non-running machines to high RUL (won't be scheduled)
                for i in range(len(self.machines)):
                    if i not in running_indices:
                        predicted_ruls[i] = 99999
                
                # Step 3: Schedule maintenance (HiGHS MILP optimizer)
                decisions = highs_scheduler(
                    predicted_ruls,
                    self.config.capacity,
                    self.config.safety_threshold,
                    self.config
                )
                
                # Step 4: Execute maintenance decisions
                for i in range(len(self.machines)):
                    if decisions[i] == 1 and self.machines[i].status == 'running':
                        true_rul = self.machines[i].true_rul
                        cost = calculate_cost('maintain', true_rul, self.config)
                        day_cost += cost
                        day_maintenances += 1
                        self.machines[i].status = 'maintained'
            
            # Step 5: Reset failed and maintained machines
            for m in self.machines:
                if m.status in ('failed', 'maintained'):
                    reset_seed_counter += 1
                    m.reset(seed=reset_seed_counter)
            
            # Record
            day_cost = round(day_cost, 2)
            self.daily_costs.append(day_cost)
            self.total_cost += day_cost
            self.total_failures += day_failures
            self.total_maintenances += day_maintenances
            
            if verbose and (day + 1) % 100 == 0:
                print(
                    f"  Day {day+1}/{n_days} | "
                    f"Day Cost: {day_cost:.1f} | "
                    f"Failures: {day_failures} | "
                    f"Maintenances: {day_maintenances} | "
                    f"Cumulative Cost: {self.total_cost:.1f}"
                )
        
        return self.get_results()
    
    def get_results(self):
        """Get simulation summary."""
        return {
            'total_cost': self.total_cost,
            'total_failures': self.total_failures,
            'total_maintenances': self.total_maintenances,
            'avg_daily_cost': self.total_cost / max(self.day_count, 1),
            'failure_rate': self.total_failures / max(self.day_count, 1),
            'maintenance_rate': self.total_maintenances / max(self.day_count, 1),
            'daily_costs': np.array(self.daily_costs),
            'n_days': self.day_count,
        }
