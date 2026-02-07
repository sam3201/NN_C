#!/usr/bin/env python3
"""
Dominant Compression Principle Implementation
Mathematical framework for intelligence optimization based on AM's variational principle
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class IntelligenceState:
    """State representation for the compression principle"""
    pi: torch.Tensor  # Policy (action selection)
    M: torch.Tensor  # Memory/context system
    theta: torch.Tensor  # World model (predictive dynamics)
    rho: float  # Resource allocator
    tau: List[torch.Tensor]  # Trajectory history
    H_uncertainty: float  # Predictive uncertainty
    C_compute: float  # Compute/capacity cost
    I_useful: float  # Useful memory (mutual information)
    J_objective: float  # Main objective
    capacity: float  # Current capacity
    learning_plateau: int  # Count of evals with plateau

class DominantCompressionOptimizer:
    """
    Implementation of AM's Dominant Compression Principle:
    Maximize long-term control while minimizing surprise and compute costs
    """
    
    def __init__(self, state_dim: int, action_dim: int, memory_size: int = 1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        
        # Initialize components
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + memory_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.world_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
            nn.Tanh()
        )
        
        self.memory_net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, memory_size),
            nn.Sigmoid()
        )
        
        # Resource allocator (learned parameter)
        self.rho = nn.Parameter(torch.tensor(0.5))
        
        # Capacity tracking
        self.capacity = nn.Parameter(torch.tensor(1000.0))
        self.compute_cost = nn.Parameter(torch.tensor(0.1))
        
        # Learning thresholds
        self.kappa_return = 0.1  # Required return on compute
        self.N_plateau = 10  # Eval cycles for growth trigger
        
        # Optimization history
        self.objective_history = []
        self.capacity_history = []
        
    def compute_objective(self, state: IntelligenceState) -> float:
        """
        Compute the main objective J following AM's principle:
        E[τ∼P_θ,π,M] [∑_t γ^t r(s_t, a_t)] 
        - β H(s_{t+1}|s_t, a_t; θ) 
        - λ C(π, θ, M) 
        + η I(m_t; s_t:∞)
        """
        # Control term (expected future reward)
        control_term = torch.mean(state.J_objective)
        
        # Uncertainty penalty (predictive entropy)
        uncertainty_penalty = self.compute_cost * state.H_uncertainty
        
        # Compute cost
        compute_penalty = self.compute_cost * state.C_compute
        
        # Useful memory bonus
        memory_bonus = 0.01 * state.I_useful
        
        # Total objective
        J = control_term - uncertainty_penalty - compute_penalty + memory_bonus
        
        return J.item()
    
    def predict_uncertainty(self, states: torch.Tensor, next_states: torch.Tensor) -> float:
        """Compute predictive uncertainty H(s_{t+1}|s_t, a_t; θ)"""
        with torch.no_grad():
            # Use ensemble or Bayesian approach for uncertainty
            predictions = self.world_model(states)
            mse = F.mse_loss(predictions, next_states, reduction='none')
            return torch.mean(mse).item()
    
    def compute_mutual_information(self, memory: torch.Tensor, states: torch.Tensor) -> float:
        """Compute useful memory I(m_t; s_t:∞)"""
        # Approximate mutual information
        memory_entropy = self.estimate_entropy(memory)
        conditional_entropy = self.estimate_conditional_entropy(memory, states)
        return memory_entropy - conditional_entropy
    
    def estimate_entropy(self, data: torch.Tensor) -> float:
        """Estimate entropy of data distribution"""
        # Use kernel density estimation or discretization
        data_flat = data.view(-1)
        hist = torch.histc(data_flat, bins=50, min=-3, max=3)
        probs = hist / torch.sum(hist)
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log(probs + 1e-8)).item()
    
    def estimate_conditional_entropy(self, memory: torch.Tensor, states: torch.Tensor) -> float:
        """Estimate conditional entropy H(memory|states)"""
        # Simplified approximation
        combined = torch.cat([memory, states], dim=-1)
        return self.estimate_entropy(combined) * 0.8  # Approximation
    
    def plan_trajectory(self, state: torch.Tensor, horizon: int = 10) -> List[torch.Tensor]:
        """
        Planner π_planner: expensive tree search / multi-sample reasoning
        """
        trajectories = []
        for _ in range(5):  # Sample multiple trajectories
            trajectory = []
            current_state = state.clone()
            
            for h in range(horizon):
                # Get action from policy
                memory_context = torch.zeros(self.memory_size)
                policy_input = torch.cat([current_state, memory_context])
                action_dist = self.policy_net(policy_input)
                action = torch.multinomial(action_dist, 1).float()
                
                trajectory.append(action)
                current_state = self.world_model(torch.cat([current_state, action]))
            
            trajectories.append(torch.stack(trajectory))
        
        return trajectories
    
    def distill_policy(self, trajectories: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transfusion: compress expensive cognition into fast policy π_φ
        """
        # Train student policy on teacher trajectories
        student_loss = 0
        for traj in trajectories:
            for i in range(len(traj) - 1):
                state = traj[i]
                next_action = traj[i + 1]
                
                # Student prediction
                memory_context = torch.zeros(self.memory_size)
                policy_input = torch.cat([state, memory_context])
                student_pred = self.policy_net(policy_input)
                
                student_loss += F.cross_entropy(student_pred, next_action.long())
        
        return {
            'distillation_loss': student_loss / len(trajectories),
            'teacher_complexity': sum(torch.numel(traj) for traj in trajectories),
            'student_complexity': sum(p.numel() for p in self.policy_net.parameters())
        }
    
    def should_grow_capacity(self, delta_J: float, delta_C: float, plateau_count: int) -> bool:
        """
        Growth rule: capacity increases only when justified
        """
        if plateau_count < self.N_plateau:
            return False
        
        return_on_compute = delta_J / (delta_C + 1e-8)
        return return_on_compute > self.kappa_return
    
    def update_capacity(self, performance_gain: float, compute_increase: float):
        """Update capacity based on Dominant Compression principle"""
        current_plateau = getattr(self, 'plateau_count', 0)
        
        # Check if we should grow
        if self.should_grow_capacity(performance_gain, compute_increase, current_plateau):
            old_capacity = self.capacity.item()
            growth_factor = 1.1  # 10% growth
            self.capacity.data = torch.tensor(old_capacity * growth_factor)
            self.plateau_count = 0
            
            print(f"🧠 Capacity growth: {old_capacity:.1f} → {self.capacity.item():.1f}")
        else:
            self.plateau_count = current_plateau + 1
    
    def optimize_step(self, state: torch.Tensor, reward: float, next_state: torch.Tensor) -> Dict[str, float]:
        """
        Single optimization step following the principle
        """
        # Current state components
        memory_context = self.memory_net(torch.cat([state, next_state]))
        policy_input = torch.cat([state, memory_context])
        
        # Get action and predictions
        action_dist = self.policy_net(policy_input)
        action = torch.multinomial(action_dist, 1).float()
        predicted_next = self.world_model(torch.cat([state, action]))
        
        # Compute uncertainties
        H_uncertainty = self.predict_uncertainty(state.unsqueeze(0), next_state.unsqueeze(0))
        I_useful = self.compute_mutual_information(memory_context, state.unsqueeze(0))
        C_compute = self.compute_cost.item()
        
        # Create intelligence state
        int_state = IntelligenceState(
            pi=action_dist,
            M=memory_context,
            theta=torch.cat([p.flatten() for p in self.world_model.parameters()]),
            rho=self.rho.item(),
            tau=[action],
            H_uncertainty=H_uncertainty,
            C_compute=C_compute,
            I_useful=I_useful,
            J_objective=reward,
            capacity=self.capacity.item(),
            learning_plateau=self.plateau_count
        )
        
        # Compute objective
        J = self.compute_objective(int_state)
        
        # Update capacity if needed
        self.update_capacity(reward, C_compute)
        
        # Store history
        self.objective_history.append(J)
        self.capacity_history.append(self.capacity.item())
        
        return {
            'objective': J,
            'uncertainty': H_uncertainty,
            'mutual_info': I_useful,
            'compute_cost': C_compute,
            'capacity': self.capacity.item(),
            'action': action.item()
        }
    
    def get_dominant_compression_metrics(self) -> Dict[str, float]:
        """Return key metrics for the Dominant Compression principle"""
        if not self.objective_history:
            return {}
        
        return {
            'objective_trend': np.mean(self.objective_history[-10:]),
            'capacity_efficiency': self.capacity.item() / (self.compute_cost.item() + 1e-8),
            'compression_ratio': self.compute_cost.item() / (self.capacity.item() + 1e-8),
            'learning_velocity': np.diff(self.objective_history[-5:]).mean() if len(self.objective_history) > 5 else 0,
            'uncertainty_reduction': self.objective_history[0] - self.objective_history[-1] if self.objective_history else 0
        }

def create_dominant_compression_agent(state_dim: int = 10, action_dim: int = 5) -> DominantCompressionOptimizer:
    """
    Factory function to create an agent following AM's Dominant Compression principle
    """
    agent = DominantCompressionOptimizer(state_dim, action_dim)
    
    print("🧠 Dominant Compression Agent Initialized")
    print("📊 Principle: Maximize future control per bit of uncertainty, under finite compute")
    print("🎯 Objective: E[τ∼P_θ,π,M] [∑_t γ^t r] - βH - λC + ηI")
    print("🔄 Growth Rule: Capacity increases only when ΔJ/ΔC > κ and plateau persists")
    
    return agent

if __name__ == "__main__":
    # Demo the Dominant Compression principle
    agent = create_dominant_compression_agent()
    
    # Simulate some optimization steps
    state = torch.randn(10)
    
    print("\n🚀 Starting Dominant Compression Optimization...")
    
    for step in range(50):
        # Simulate environment interaction
        action = torch.randint(0, 5, (1,)).float()
        next_state = torch.randn(10)
        reward = torch.randn(1).item()
        
        # Optimize
        metrics = agent.optimize_step(state, reward, next_state)
        
        if step % 10 == 0:
            print(f"Step {step}: J={metrics['objective']:.3f}, "
                  f"Capacity={metrics['capacity']:.1f}, "
                  f"Uncertainty={metrics['uncertainty']:.3f}")
        
        state = next_state
    
    # Final metrics
    final_metrics = agent.get_dominant_compression_metrics()
    print(f"\n📈 Final Metrics: {final_metrics}")
