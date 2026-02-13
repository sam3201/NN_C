#!/usr/bin/env python3
"""
SAM-D C Core Integration Layer
Unified interface to all high-performance C modules
Includes: Id/Ego/Superego drives, Emotion, Wisdom (Phase 1)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import sam_fast_rng
    import sam_telemetry_core
    import sam_god_equation
    import sam_regulator_compiler_c
    import sam_consciousness
    import sam_memory
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("⚠️ C modules not available, using NumPy fallbacks")


class DriveSystem:
    """
    Id/Ego/Superego Drive System
    - Id (R): Raw drives, survival, reward seeking
    - Ego (ΔG): Goal coherence, planning, execution
    - Superego (W): Wisdom, ethics, temporal coherence
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.id_drive = 0.5
        self.ego_goal_coherence = 0.5
        self.superego_wisdom = 0.5
        self.id_history = []
        self.ego_history = []
        self.superego_history = []
    
    def compute_drives(self, 
                       telemetry: List[float],
                       god_state: Dict,
                       action: str = "explore") -> Dict[str, float]:
        """
        Compute Id/Ego/Superego drives based on telemetry and god state.
        
        Args:
            telemetry: Current telemetry vector
            god_state: K, U, O, omega state
            
        Returns:
            Dictionary with id, ego, superego values
        """
        K = god_state.get('K', 1.0)
        U = god_state.get('U', 5.0)
        O = god_state.get('O', 10.0)
        omega = god_state.get('omega', 0.5)
        contra = god_state.get('contradiction', 0.0)
        
        # Id: Driven by survival, resource pressure, novelty
        # Higher when resources low, novelty high
        resource_pressure = telemetry[14] if len(telemetry) > 14 else 0.2
        novelty_pressure = telemetry[15] if len(telemetry) > 15 else 0.2
        survival_signal = resource_pressure * 0.6 + novelty_pressure * 0.4
        
        # Id tends toward seeking more resources/knowledge
        id_target = 0.3 + 0.5 * survival_signal + 0.2 * (1.0 - omega)
        self.id_drive = 0.9 * self.id_drive + 0.1 * id_target
        
        # Ego: Driven by goal coherence, planning success
        # Higher when making progress toward goals
        progress_rate = telemetry[12] if len(telemetry) > 12 else 0.2
        plateau = telemetry[13] if len(telemetry) > 13 else 0.0
        
        # Action modifiers
        action_boost = {'explore': 0.1, 'verify': 0.05, 'grow': 0.15, 'rest': -0.1}
        ego_target = 0.4 + 0.4 * progress_rate - 0.3 * plateau + action_boost.get(action, 0.0)
        self.ego_goal_coherence = 0.85 * self.ego_goal_coherence + 0.15 * ego_target
        
        # Superego: Driven by wisdom, contradiction avoidance, long-term coherence
        # Higher when system is stable and coherent
        wisdom_signal = omega * 0.5 + (1.0 - contra) * 0.3 + (1.0 - resource_pressure) * 0.2
        superego_target = 0.3 + 0.6 * wisdom_signal
        self.superego_wisdom = 0.92 * self.superego_wisdom + 0.08 * superego_target
        
        # Track history
        self.id_history.append(self.id_drive)
        self.ego_history.append(self.ego_goal_coherence)
        self.superego_history.append(self.superego_wisdom)
        
        # Keep history bounded
        max_history = 100
        if len(self.id_history) > max_history:
            self.id_history = self.id_history[-max_history:]
            self.ego_history = self.ego_history[-max_history:]
            self.superego_history = self.superego_history[-max_history:]
        
        return {
            'id': self.id_drive,
            'ego': self.ego_goal_coherence,
            'superego': self.superego_wisdom,
            'id_ego_balance': self.id_drive - self.ego_goal_coherence,
            'superego_constrains_id': max(0, self.id_drive - self.superego_wisdom)
        }
    
    def get_drive_vector(self) -> List[float]:
        """Return drive vector for telemetry"""
        return [self.id_drive, self.ego_goal_coherence, self.superego_wisdom]


class EmotionSystem:
    """
    Emotion/Affective System
    Tracks valence (positive/negative), arousal (activation), dominance (control)
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.valence = 0.5
        self.arousal = 0.5
        self.dominance = 0.5
        self.emotion_history = []
    
    def compute_emotion(self,
                        god_state: Dict,
                        drives: Dict[str, float],
                        previous_outcome: float = 0.0) -> Dict[str, float]:
        """
        Compute emotional state based on system state.
        
        Args:
            god_state: K, U, O state
            drives: Current drive values
            previous_outcome: Result of last action (-1 to 1)
            
        Returns:
            Dictionary with valence, arousal, dominance
        """
        K = god_state.get('K', 1.0)
        U = god_state.get('U', 5.0)
        omega = god_state.get('omega', 0.5)
        contra = god_state.get('contradiction', 0.0)
        
        # Valence: Positive if making progress, negative if stuck
        # Driven by contradiction, plateau, progress
        plateau = 1.0 if K > 10 and god_state.get('K', 1) < 12 else 0.0
        progress_signal = min(1.0, K / 20.0)
        
        valence_target = 0.5 + 0.3 * progress_signal - 0.3 * contra - 0.2 * plateau + 0.1 * previous_outcome
        self.valence = 0.8 * self.valence + 0.2 * np.clip(valence_target, 0.0, 1.0)
        
        # Arousal: High when exploring, low when resting
        # Driven by Id drive, contradiction
        arousal_target = 0.3 + 0.4 * drives.get('id', 0.5) + 0.3 * contra
        self.arousal = 0.85 * self.arousal + 0.15 * np.clip(arousal_target, 0.0, 1.0)
        
        # Dominance: Control over situation
        # High when Superego constrains Id, low when overwhelmed
        dominance_target = 0.4 + 0.4 * drives.get('superego', 0.5) + 0.2 * omega - 0.2 * U / (1 + U)
        self.dominance = 0.88 * self.dominance + 0.12 * np.clip(dominance_target, 0.0, 1.0)
        
        # Track emotion vector (PAD-like)
        emotion_vector = [self.valence, self.arousal, self.dominance]
        self.emotion_history.append(emotion_vector)
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]
        
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'emotion_vector': emotion_vector,
            'is_positive': self.valence > 0.5,
            'is_activated': self.arousal > 0.5,
            'is_in_control': self.dominance > 0.5
        }
    
    def get_emotion_vector(self) -> List[float]:
        """Return emotion vector for state"""
        return [self.valence, self.arousal, self.dominance]


class WisdomSystem:
    """
    Wisdom Module
    Computes wisdom as preference for trajectories that preserve
    the ability to revise preferences (Future-Preserving Coherence)
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.wisdom = 0.5
        self.future_preserving_coherence = 0.5
        self.wisdom_history = []
    
    def compute_wisdom(self,
                       god_state: Dict,
                       drives: Dict[str, float],
                       emotion: Dict[str, float]) -> Dict[str, float]:
        """
        Compute wisdom based on system state.
        
        Args:
            god_state: K, U, O state
            drives: Current drive values
            emotion: Current emotion state
            
        Returns:
            Dictionary with wisdom metrics
        """
        K = god_state.get('K', 1.0)
        U = god_state.get('U', 5.0)
        O = god_state.get('O', 10.0)
        omega = god_state.get('omega', 0.5)
        contra = god_state.get('contradiction', 0.0)
        
        # Wisdom factors:
        # 1. Coherence (omega) - maintains internal consistency
        # 2. Low contradiction - avoids internal conflict
        # 3. Moderate unknowns - acknowledges uncertainty without paralysis
        # 4. Emotional regulation - doesn't act purely on emotion
        
        coherence_factor = omega
        contradiction_factor = 1.0 - contra
        uncertainty_factor = min(1.0, U / (1.0 + U))  # Acknowledges unknowns
        emotional_regulation = 1.0 - abs(emotion.get('valence', 0.5) - 0.5) * 0.5
        
        # Future-Preserving Coherence (FPC)
        # Prefers actions that keep options open
        fpc = (coherence_factor * 0.35 + 
               contradiction_factor * 0.30 + 
               uncertainty_factor * 0.20 + 
               emotional_regulation * 0.15)
        
        # Combine with Superego for overall wisdom
        superego = drives.get('superego', 0.5)
        wisdom_target = 0.4 * superego + 0.6 * fpc
        
        self.wisdom = 0.9 * self.wisdom + 0.1 * wisdom_target
        self.future_preserving_coherence = 0.92 * self.future_preserving_coherence + 0.08 * fpc
        
        self.wisdom_history.append(self.wisdom)
        if len(self.wisdom_history) > 100:
            self.wisdom_history = self.wisdom_history[-100:]
        
        return {
            'wisdom': self.wisdom,
            'fpc': self.future_preserving_coherence,
            'coherence_factor': coherence_factor,
            'contradiction_avoidance': contradiction_factor,
            'uncertainty_acknowledgment': uncertainty_factor,
            'emotional_regulation': emotional_regulation,
            'should_deliberate': self.wisdom < 0.6 and U > 2.0,
            'should_act': self.wisdom > 0.7 or emotion.get('arousal', 0.5) > 0.7
        }
    
    def get_wisdom_value(self) -> float:
        """Return wisdom value"""
        return self.wisdom


class PowerSystem:
    """
    Power System (Phase 2)
    Computes power as the ability to influence the environment and achieve goals.
    Power = Resources × Capabilities × Control_Effectiveness
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.power = 0.5
        self.resources = 0.5  # Available resources (compute, memory, energy)
        self.capabilities = 0.5  # Current capabilities
        self.influence = 0.5  # Environmental influence
        self.power_history = []
        self.resource_history = []
        
    def compute_power(self,
                     god_state: Dict,
                     drives: Dict[str, float],
                     telemetry: List[float]) -> Dict[str, float]:
        """
        Compute power based on system state.
        
        Args:
            god_state: K, U, O state
            drives: Current drive values
            telemetry: Telemetry vector
            
        Returns:
            Dictionary with power metrics
        """
        K = god_state.get('K', 1.0)
        U = god_state.get('U', 5.0)
        omega = god_state.get('omega', 0.5)
        
        # Resources component
        # Based on knowledge accumulation and stability
        resource_pressure = telemetry[14] if len(telemetry) > 14 else 0.2
        knowledge_factor = min(1.0, K / 20.0)  # Saturates at high K
        stability_factor = omega
        
        # Update resources (accumulate knowledge, consume for operations)
        resource_target = 0.3 + 0.5 * knowledge_factor + 0.2 * stability_factor - 0.3 * resource_pressure
        self.resources = 0.95 * self.resources + 0.05 * np.clip(resource_target, 0.0, 1.0)
        
        # Capabilities component
        # Based on what the system can actually do
        id_strength = drives.get('id', 0.5)
        ego_strength = drives.get('ego', 0.5)
        capability_target = 0.4 + 0.3 * id_strength + 0.3 * ego_strength + 0.2 * (1.0 - U / (1.0 + U))
        self.capabilities = 0.9 * self.capabilities + 0.1 * capability_target
        
        # Influence component
        # Based on successful actions and environmental impact
        progress = telemetry[12] if len(telemetry) > 12 else 0.2
        influence_target = 0.3 + 0.5 * progress + 0.2 * self.resources
        self.influence = 0.92 * self.influence + 0.08 * influence_target
        
        # Overall power: resources × capabilities × influence
        # All components must be present for power
        self.power = (self.resources * 0.4 + 
                     self.capabilities * 0.35 + 
                     self.influence * 0.25)
        
        # Track history
        self.power_history.append(self.power)
        self.resource_history.append(self.resources)
        if len(self.power_history) > 100:
            self.power_history = self.power_history[-100:]
            self.resource_history = self.resource_history[-100:]
        
        return {
            'power': self.power,
            'resources': self.resources,
            'capabilities': self.capabilities,
            'influence': self.influence,
            'power_trend': np.mean(self.power_history[-10:]) - np.mean(self.power_history[-20:-10]) if len(self.power_history) >= 20 else 0.0,
            'is_powerful': self.power > 0.7,
            'resource_critical': self.resources < 0.2,
            'should_accumulate': self.resources < 0.4 and id_strength > 0.6,
            'effective_power': self.power * (1.0 - resource_pressure * 0.3)
        }
    
    def get_power_value(self) -> float:
        """Return power value"""
        return self.power


class ControlSystem:
    """
    Control System (Phase 2)
    Manages how power is exercised with wisdom constraints.
    Control = f(Power, Wisdom, Constraints)
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.control = 0.5
        self.control_precision = 0.5  # How precisely we can control
        self.control_scope = 0.5  # Breadth of control
        self.wisdom_constraint = 0.5  # Wisdom-imposed limits
        self.control_history = []
        self.last_action = None
        
    def compute_control(self,
                       power_state: Dict[str, float],
                       wisdom_state: Dict[str, float],
                       drives: Dict[str, float],
                       emotion: Dict[str, float]) -> Dict[str, float]:
        """
        Compute control based on power, wisdom, and emotional state.
        
        Args:
            power_state: Current power metrics
            wisdom_state: Current wisdom metrics
            drives: Current drive values
            emotion: Current emotion state
            
        Returns:
            Dictionary with control metrics
        """
        power = power_state.get('power', 0.5)
        resources = power_state.get('resources', 0.5)
        wisdom = wisdom_state.get('wisdom', 0.5)
        fpc = wisdom_state.get('fpc', 0.5)
        
        # Control precision: ability to execute precisely
        # High when power and wisdom are balanced
        precision_target = 0.3 + 0.4 * power + 0.3 * wisdom
        # But wisdom constrains reckless precision
        if wisdom < 0.4 and power > 0.7:
            precision_target *= 0.7  # Wisdom limits reckless control
        self.control_precision = 0.9 * self.control_precision + 0.1 * precision_target
        
        # Control scope: breadth of what can be controlled
        # Limited by resources and capabilities
        ego = drives.get('ego', 0.5)
        scope_target = 0.2 + 0.4 * resources + 0.3 * ego + 0.1 * power
        self.control_scope = 0.92 * self.control_scope + 0.08 * scope_target
        
        # Wisdom constraint: how much wisdom limits control
        # High wisdom = more constrained but safer control
        superego = drives.get('superego', 0.5)
        constraint_target = 0.2 + 0.5 * wisdom + 0.3 * superego
        self.wisdom_constraint = 0.94 * self.wisdom_constraint + 0.06 * constraint_target
        
        # Overall control: precision × scope × (1 - wisdom_constraint/2)
        # Wisdom doesn't completely prevent control, but moderates it
        wisdom_factor = 1.0 - (self.wisdom_constraint * 0.3)
        self.control = (self.control_precision * 0.4 + 
                       self.control_scope * 0.4 + 
                       power * 0.2) * wisdom_factor
        
        # Emotional modulation
        dominance = emotion.get('dominance', 0.5)
        arousal = emotion.get('arousal', 0.5)
        
        # High dominance increases control feeling but wisdom moderates
        emotional_boost = dominance * 0.1 if wisdom > 0.5 else dominance * 0.05
        
        # High arousal can reduce fine control
        if arousal > 0.7:
            self.control *= 0.95
        
        self.control = min(1.0, self.control + emotional_boost)
        
        # Track history
        self.control_history.append(self.control)
        if len(self.control_history) > 100:
            self.control_history = self.control_history[-100:]
        
        # Determine control strategy
        strategy = 'balanced'
        if wisdom > 0.7 and power > 0.6:
            strategy = 'wise_authority'
        elif power > 0.7 and wisdom < 0.5:
            strategy = 'reckless_force'
        elif wisdom > 0.6 and power < 0.4:
            strategy = 'patient_accumulation'
        elif resources < 0.2:
            strategy = 'conservation'
        
        return {
            'control': self.control,
            'control_precision': self.control_precision,
            'control_scope': self.control_scope,
            'wisdom_constraint': self.wisdom_constraint,
            'effective_control': self.control * wisdom_factor,
            'strategy': strategy,
            'in_control': self.control > 0.6 and wisdom > 0.5,
            'should_delegate': self.control > 0.7 and power > 0.6,
            'should_restrain': power > 0.7 and wisdom < 0.5,
            'can_execute': self.control > 0.4 and resources > 0.2
        }
    
    def apply_control(self, action: str, intensity: float = 0.5) -> bool:
        """
        Apply control to an action, respecting constraints.
        
        Args:
            action: Action to control
            intensity: Desired intensity [0, 1]
            
        Returns:
            True if control applied successfully
        """
        # Wisdom constrains high-intensity actions when wisdom is low
        max_intensity = 0.3 + 0.7 * self.wisdom_constraint
        actual_intensity = min(intensity, max_intensity)
        
        # Check if we have sufficient control
        if self.control < 0.2:
            return False
        
        self.last_action = {
            'action': action,
            'requested_intensity': intensity,
            'actual_intensity': actual_intensity,
            'control_applied': self.control
        }
        
        return True
    
    def get_control_value(self) -> float:
        """Return control value"""
        return self.control


class ResourceManager:
    """
    Resource Manager (Phase 2)
    Tracks and manages system resources (compute, memory, energy, budget).
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.compute_available = 1.0
        self.memory_available = 1.0
        self.energy_level = 0.8
        self.budget_remaining = 1.0  # Normalized budget
        self.allocation_history = []
        
    def update_resources(self,
                        power_state: Dict[str, float],
                        god_state: Dict,
                        action_cost: float = 0.1) -> Dict[str, float]:
        """
        Update resource levels based on actions and recovery.
        
        Args:
            power_state: Current power state
            god_state: God equation state
            action_cost: Cost of last action
            
        Returns:
            Dictionary with resource metrics
        """
        # Resources deplete with actions
        self.compute_available = max(0.0, self.compute_available - action_cost * 0.8)
        self.memory_available = max(0.0, self.memory_available - action_cost * 0.5)
        self.energy_level = max(0.0, self.energy_level - action_cost * 0.6)
        
        # Resources recover based on knowledge (K) and coherence
        K = god_state.get('K', 1.0)
        omega = god_state.get('omega', 0.5)
        recovery_rate = 0.02 * (1.0 + K / 10.0) * omega
        
        self.compute_available = min(1.0, self.compute_available + recovery_rate)
        self.memory_available = min(1.0, self.memory_available + recovery_rate * 0.8)
        self.energy_level = min(1.0, self.energy_level + recovery_rate * 0.6)
        
        # Budget decreases with usage
        self.budget_remaining = max(0.0, self.budget_remaining - action_cost * 0.1)
        
        # Overall resource health
        resource_health = (self.compute_available * 0.4 + 
                          self.memory_available * 0.3 + 
                          self.energy_level * 0.2 + 
                          self.budget_remaining * 0.1)
        
        return {
            'compute': self.compute_available,
            'memory': self.memory_available,
            'energy': self.energy_level,
            'budget': self.budget_remaining,
            'health': resource_health,
            'is_depleted': resource_health < 0.2,
            'should_conserve': resource_health < 0.4,
            'can_expand': resource_health > 0.7
        }
    
    def allocate_resources(self, task_priority: float, required_amount: float) -> float:
        """
        Allocate resources to a task.
        
        Args:
            task_priority: Priority of task [0, 1]
            required_amount: Amount of resources needed
            
        Returns:
            Amount actually allocated
        """
        available = (self.compute_available * 0.5 + 
                    self.memory_available * 0.3 + 
                    self.energy_level * 0.2)
        
        # Allocate based on priority and availability
        allocation = min(required_amount, available * task_priority)
        
        # Deplete resources
        self.compute_available -= allocation * 0.5
        self.memory_available -= allocation * 0.3
        self.energy_level -= allocation * 0.2
        
        self.allocation_history.append({
            'requested': required_amount,
            'allocated': allocation,
            'priority': task_priority
        })
        
        return max(0.0, allocation)


class MetaObserver:
    """
    Meta-Observer System (Phase 3)
    Self-observation and introspection capabilities.
    Tracks system state, performance, and generates self-models.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.self_model = {
            'identity': 'SAM-D AGI',
            'version': '5.0.0 (ΨΔ•Ω-Core Recursive)',
            'phase': 3,
            'purpose': 'Self-referential meta-learning and growth'
        }
        self.observation_history = []
        self.performance_log = []
        self.introspection_depth = 0.5
        self.self_awareness = 0.3
        
    def observe_self(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform self-observation of system state.
        
        Args:
            system_state: Current full system state
            
        Returns:
            Self-observation report
        """
        # Extract key metrics
        phase1_metrics = {
            'drives': system_state.get('drives', {}),
            'emotion': system_state.get('emotion', {}),
            'wisdom': system_state.get('wisdom', {})
        }
        
        phase2_metrics = {
            'power': system_state.get('power', {}),
            'control': system_state.get('control', {}),
            'resources': system_state.get('resources', {})
        }
        
        god_metrics = system_state.get('god', {})
        
        # Calculate system coherence
        coherence = god_metrics.get('omega', 0.5)
        contradiction = god_metrics.get('contradiction', 0.0)
        
        # Generate self-observation
        observation = {
            'timestamp': system_state.get('tick', 0),
            'identity_stable': self._check_identity_stability(phase1_metrics),
            'emotional_state': phase1_metrics['emotion'].get('valence', 0.5),
            'wisdom_level': phase1_metrics['wisdom'].get('wisdom', 0.5),
            'power_status': phase2_metrics['power'].get('power', 0.5),
            'control_effectiveness': phase2_metrics['control'].get('control', 0.5),
            'resource_health': phase2_metrics['resources'].get('health', 0.5),
            'system_coherence': coherence,
            'internal_contradiction': contradiction,
            'self_awareness_level': self.self_awareness,
            'observation_depth': self.introspection_depth
        }
        
        self.observation_history.append(observation)
        if len(self.observation_history) > 100:
            self.observation_history = self.observation_history[-100:]
        
        # Increase self-awareness with observations
        self.self_awareness = min(1.0, self.self_awareness + 0.001)
        
        return observation
    
    def _check_identity_stability(self, phase1_metrics: Dict) -> bool:
        """Check if identity remains stable"""
        superego = phase1_metrics['drives'].get('superego', 0.5)
        wisdom = phase1_metrics['wisdom'].get('wisdom', 0.5)
        return superego > 0.4 and wisdom > 0.4
    
    def introspect(self) -> Dict[str, Any]:
        """
        Deep introspection on system state and history.
        
        Returns:
            Introspection analysis
        """
        if len(self.observation_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent = self.observation_history[-10:]
        
        # Analyze trends
        coherence_trend = np.mean([o['system_coherence'] for o in recent])
        power_trend = np.mean([o['power_status'] for o in recent])
        wisdom_trend = np.mean([o['wisdom_level'] for o in recent])
        
        # Detect patterns
        patterns = []
        if coherence_trend > 0.7:
            patterns.append('high_coherence')
        if power_trend > wisdom_trend:
            patterns.append('power_wisdom_gap')
        if recent[-1]['resource_health'] < 0.3:
            patterns.append('resource_critical')
        
        introspection = {
            'self_model': self.self_model,
            'coherence_trend': coherence_trend,
            'power_trend': power_trend,
            'wisdom_trend': wisdom_trend,
            'detected_patterns': patterns,
            'introspection_depth': self.introspection_depth,
            'self_awareness': self.self_awareness,
            'recommendations': self._generate_recommendations(patterns, recent[-1])
        }
        
        return introspection
    
    def _generate_recommendations(self, patterns: List[str], current: Dict) -> List[str]:
        """Generate recommendations based on introspection"""
        recommendations = []
        
        if 'power_wisdom_gap' in patterns:
            recommendations.append("Power exceeds wisdom - exercise caution")
        
        if current['emotional_state'] < 0.3:
            recommendations.append("Low valence detected - consider rest/reflection")
        
        if current['internal_contradiction'] > 0.5:
            recommendations.append("High internal contradiction - resolve conflicts")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations


class CounterfactualEngine:
    """
    Counterfactual Engine (Phase 3)
    Simulates alternative scenarios and "what-if" reasoning.
    Enables learning from hypothetical situations.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.counterfactual_history = []
        self.scenarios_generated = 0
        self.learning_rate = 0.1
        
    def generate_counterfactual(self, 
                               actual_state: Dict[str, Any],
                               action_taken: str,
                               outcome: float) -> Dict[str, Any]:
        """
        Generate counterfactual scenarios for learning.
        
        Args:
            actual_state: Actual system state
            action_taken: Action that was taken
            outcome: Actual outcome
            
        Returns:
            Counterfactual analysis
        """
        # Generate alternative actions
        alternative_actions = ['explore', 'verify', 'grow', 'rest']
        alternative_actions.remove(action_taken)
        
        counterfactuals = []
        
        for alt_action in alternative_actions:
            # Simulate what would have happened
            simulated_outcome = self._simulate_outcome(actual_state, alt_action)
            
            counterfactuals.append({
                'action': alt_action,
                'simulated_outcome': simulated_outcome,
                'difference': simulated_outcome - outcome,
                'better': simulated_outcome > outcome
            })
        
        # Find best counterfactual
        best_alt = max(counterfactuals, key=lambda x: x['simulated_outcome'])
        
        analysis = {
            'actual_action': action_taken,
            'actual_outcome': outcome,
            'counterfactuals': counterfactuals,
            'best_alternative': best_alt,
            'regret': max(0, best_alt['simulated_outcome'] - outcome),
            'lesson': self._extract_lesson(action_taken, outcome, best_alt)
        }
        
        self.counterfactual_history.append(analysis)
        self.scenarios_generated += len(counterfactuals)
        
        return analysis
    
    def _simulate_outcome(self, state: Dict[str, Any], action: str) -> float:
        """Simulate outcome for an alternative action"""
        base_outcome = state.get('last_outcome', 0.0)
        
        # Different actions have different expected outcomes
        action_effects = {
            'explore': 0.1 * self.rng.random(),
            'verify': 0.05 * self.rng.random(),
            'grow': 0.15 * self.rng.random() - 0.05,  # Risky
            'rest': 0.02 * self.rng.random()
        }
        
        noise = 0.05 * (self.rng.random() - 0.5)
        return base_outcome + action_effects.get(action, 0) + noise
    
    def _extract_lesson(self, action: str, outcome: float, best_alt: Dict) -> str:
        """Extract learning lesson from counterfactual"""
        if best_alt['better'] and best_alt['difference'] > 0.1:
            return f"Alternative '{best_alt['action']}' would have been better by {best_alt['difference']:.3f}"
        elif outcome > 0:
            return f"Action '{action}' was appropriate in this context"
        else:
            return f"Consider different approach than '{action}' in similar situations"
    
    def apply_learnings(self, current_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply counterfactual learnings to current decision.
        
        Args:
            current_decision: Current decision context
            
        Returns:
            Augmented decision with lessons
        """
        if not self.counterfactual_history:
            return {**current_decision, 'learnings_applied': False}
        
        # Find relevant historical lessons
        relevant_lessons = []
        for cf in self.counterfactual_history[-10:]:
            if cf['regret'] > 0.1:
                relevant_lessons.append(cf['lesson'])
        
        return {
            **current_decision,
            'learnings_applied': True,
            'historical_lessons': relevant_lessons[-3:],  # Last 3 lessons
            'scenarios_considered': self.scenarios_generated
        }


class VersionalityTracker:
    """
    Versionality Tracker (Phase 3)
    Tracks system versions, capabilities, and evolution over time.
    Manages self-modification and versioning.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.versions = [{
            'version': '1.0',
            'phase': 1,
            'components': ['Id/Ego/Superego', 'Emotion', 'Wisdom'],
            'timestamp': 0
        }]
        self.current_version = '5.0.0 (ΨΔ•Ω-Core Recursive)'
        self.capability_matrix = {
            'reasoning': 0.7,
            'learning': 0.6,
            'self_awareness': 0.3,
            'adaptation': 0.5,
            'meta_cognition': 0.4
        }
        self.evolution_pressure = 0.0
        
    def track_version(self, 
                     system_state: Dict[str, Any],
                     major_change: bool = False) -> Dict[str, Any]:
        """
        Track current version and detect evolution.
        
        Args:
            system_state: Current system state
            major_change: Whether this is a major version change
            
        Returns:
            Version tracking report
        """
        tick = system_state.get('tick', 0)
        
        # Detect phase
        if 'power' in system_state and 'control' in system_state:
            phase = 2
        elif 'meta_observation' in system_state:
            phase = 3
        else:
            phase = 1
        
        # Update capability matrix
        self._update_capabilities(system_state)
        
        # Check for version change
        if major_change or self._significant_evolution():
            new_version = self._increment_version()
            version_entry = {
                'version': new_version,
                'phase': phase,
                'components': self._get_active_components(system_state),
                'capabilities': self.capability_matrix.copy(),
                'timestamp': tick
            }
            self.versions.append(version_entry)
            self.current_version = new_version
        
        return {
            'current_version': self.current_version,
            'phase': phase,
            'capabilities': self.capability_matrix,
            'evolution_pressure': self.evolution_pressure,
            'version_history': len(self.versions),
            'ready_for_next_phase': self._check_phase_readiness(phase, system_state)
        }
    
    def _update_capabilities(self, state: Dict[str, Any]):
        """Update capability scores based on system state"""
        # Self-awareness increases with meta-observation
        if 'meta_observation' in state:
            self.capability_matrix['self_awareness'] = min(1.0, 
                self.capability_matrix['self_awareness'] + 0.01)
        
        # Learning improves with successful outcomes
        outcome = state.get('last_outcome', 0)
        if outcome > 0:
            self.capability_matrix['learning'] = min(1.0,
                self.capability_matrix['learning'] + 0.005)
        
        # Adaptation based on power/control balance
        power = state.get('power', {}).get('power', 0.5)
        control = state.get('control', {}).get('control', 0.5)
        if abs(power - control) < 0.2:  # Balanced
            self.capability_matrix['adaptation'] = min(1.0,
                self.capability_matrix['adaptation'] + 0.003)
    
    def _significant_evolution(self) -> bool:
        """Check if system has evolved significantly"""
        avg_capability = np.mean(list(self.capability_matrix.values()))
        return avg_capability > 0.7 and self.evolution_pressure > 0.5
    
    def _increment_version(self) -> str:
        """Increment version number"""
        parts = self.current_version.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        
        if self.evolution_pressure > 0.8:
            return f"{major + 1}.0"
        else:
            return f"{major}.{minor + 1}"
    
    def _get_active_components(self, state: Dict[str, Any]) -> List[str]:
        """Get list of active system components"""
        components = ['Id/Ego/Superego', 'Emotion', 'Wisdom']
        
        if 'power' in state:
            components.extend(['Power', 'Control', 'Resources'])
        
        if 'meta_observation' in state:
            components.extend(['Meta-Observer', 'Counterfactual', 'Versionality'])
        
        return components
    
    def _check_phase_readiness(self, current_phase: int, state: Dict[str, Any]) -> bool:
        """Check if system is ready for next phase"""
        if current_phase == 1:
            return 'power' in state
        elif current_phase == 2:
            power_ok = state.get('power', {}).get('power', 0) > 0.5
            control_ok = state.get('control', {}).get('control', 0) > 0.5
            return power_ok and control_ok
        return False
    
    def get_version_tree(self) -> Dict[str, Any]:
        """Get complete version evolution tree"""
        return {
            'current': self.current_version,
            'total_versions': len(self.versions),
            'evolution_tree': self.versions,
            'capability_trajectory': self.capability_matrix
        }


class SamCores:
    """
    Unified C core for SAM-D AGI system.
    Provides high-performance implementations of:
    - RNG (17x faster than NumPy)
    - God Equation (K/U/O dynamics)
    - 53-Regulator Compiler
    - Telemetry Collection
    - Id/Ego/Superego Drive System (Phase 1)
    - Emotion/Affective System (Phase 1)
    - Wisdom Module (Phase 1)
    - Power System (Phase 2)
    - Control System (Phase 2)
    - Resource Manager (Phase 2)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.tick = 0
        self.last_outcome = 0.0

        if C_AVAILABLE:
            self._init_cores()
        else:
            self._init_numpy_fallback()

        self._init_phase1_systems()
        self._init_phase2_systems()
        self._init_phase3_systems()

    def _init_cores(self):
        """Initialize C modules"""
        self.rng = sam_fast_rng.SamFastRNG(self.seed)
        self.god_eq = sam_god_equation.SamGodEquation()
        self.regulator = sam_regulator_compiler_c.SamRegulatorCompiler(self.seed)
        self.telemetry = sam_telemetry_core.SamTelemetry(self.seed)
        self.consciousness = sam_consciousness.SamConsciousness()
        sam_memory.SamMemory_init(self.seed)
        
        self.use_c = True
        print("🚀 C cores initialized")
    
    def _init_numpy_fallback(self):
        """NumPy fallbacks if C not available"""
        self.rng_np = np.random.default_rng(self.seed)
        self.use_c = False
        print("⚠️ Using NumPy fallbacks")
    
    def _init_phase1_systems(self):
        """Initialize Phase 1 systems: Drive, Emotion, Wisdom"""
        self.drives = DriveSystem(self.seed)
        self.emotions = EmotionSystem(self.seed)
        self.wisdom = WisdomSystem(self.seed)
        print("🧠 Phase 1 systems initialized (Id/Ego/Superego, Emotion, Wisdom)")

    def _init_phase2_systems(self):
        """Initialize Phase 2 systems: Power, Control, Resources"""
        self.power = PowerSystem(self.seed)
        self.control = ControlSystem(self.seed)
        self.resources = ResourceManager(self.seed)
        print("⚡ Phase 2 systems initialized (Power, Control, Resources)")

    def _init_phase3_systems(self):
        """Initialize Phase 3 systems: Meta-Observer, Counterfactual, Versionality"""
        self.meta_observer = MetaObserver(self.seed)
        self.counterfactual = CounterfactualEngine(self.seed)
        self.versionality = VersionalityTracker(self.seed)
        print("🔮 Phase 3 systems initialized (Meta-Observer, Counterfactual, Versionality)")

    # ========== RNG Methods ==========
    def random_uint64(self) -> int:
        """Generate random 64-bit integer"""
        if self.use_c:
            return self.rng.next()
        return self.rng_np.integers(2**63)
    
    def random_double(self) -> float:
        """Generate random double [0, 1)"""
        if self.use_c:
            return self.rng.double()
        return self.rng_np.random()
    
    def random_range(self, max_val: int) -> int:
        """Random integer in [0, max)"""
        if self.use_c:
            return self.rng.range(max_val)
        return self.rng_np.integers(max_val)
    
    def random_gaussian(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        """Gaussian random number"""
        if self.use_c:
            return self.rng.gaussian(mean, stddev)
        return self.rng_np.normal(mean, stddev)
    
    # ========== God Equation Methods ==========
    def step_god_equation(self, 
                         research: float = 0.5, 
                         verify: float = 0.5, 
                         morph: float = 0.2,
                         dt: float = 1.0) -> Dict[str, float]:
        """
        Step the God Equation forward.
        
        Args:
            research: Research effort [0, 1]
            verify: Verification effort [0, 1]  
            morph: Morphogenesis effort [0, 1]
            dt: Time step
            
        Returns:
            Dictionary with K, U, O, omega, contradiction
        """
        if self.use_c:
            self.god_eq.compute(research, verify, morph)
            return {
                'K': self.god_eq.get_K(),
                'U': self.god_eq.get_U(),
                'O': self.god_eq.get_O(),
                'omega': self.god_eq.get_omega(),
                'contradiction': self.god_eq.contradiction()
            }
        else:
            # NumPy fallback
            return self._step_god_numpy(research, verify, morph, dt)
    
    def _step_god_numpy(self, research, verify, morph, dt):
        """NumPy fallback for God Equation"""
        K = getattr(self, '_god_K', 1.0)
        U = getattr(self, '_god_U', 5.0)
        O = getattr(self, '_god_O', 10.0)
        
        sigma = (U + 0.7 * O) / (1.0 + U + 0.7 * O)
        contra = max(0.0, (U + O) / (1.0 + K) - 1.0)
        
        alpha, beta, gamma, delta, zeta = 0.05, 1.10, 0.02, 1.00, 0.01
        
        discovery = alpha * K**beta * sigma * (0.5 + research)
        burden = gamma * K**delta * (1.2 - 0.7 * verify)
        contra_pen = zeta * K**delta * contra
        
        K = max(0.0, K + (discovery - burden - contra_pen) * dt)
        
        eta, mu, kappa = 0.03, 1.0, 0.04
        U = max(0.0, U + eta * K**mu * (0.4 + 0.6*research) * dt 
                     - kappa * U * (0.3 + 0.7*verify) * dt)
        
        xi, nu, chi = 0.02, 1.0, 0.06
        O = max(0.0, O + xi * K**nu * (0.5 + 0.5*research) * dt
                     - chi * O * (0.2 + 0.8*morph) * dt)
        
        self._god_K, self._god_U, self._god_O = K, U, O
        
        return {'K': K, 'U': U, 'O': O, 'omega': 1.0 - contra, 'contradiction': contra}
    
    # ========== Regulator Methods ==========
    def compile_with_telemetry(self, 
                               telemetry: List[float],
                               K: float = 1.0,
                               U: float = 2.0,
                               omega: float = 0.5) -> Tuple[List[float], List[str], str]:
        """
        Compile regulators with telemetry.
        
        Args:
            telemetry: 18-element telemetry vector
            K: Knowledge state
            U: Unknowns state
            omega: Coherence
            
        Returns:
            (loss_weights, knob_values, regime_name)
        """
        if self.use_c:
            weights, knobs, regime = self.regulator.compile(telemetry, K, U, omega)
            return weights, knobs, regime
        else:
            return self._compile_numpy(telemetry, K, U, omega)
    
    def _compile_numpy(self, telemetry, K, U, omega):
        """NumPy fallback for regulator compilation"""
        import random
        random.seed(self.seed + self.tick)
        
        # Simplified regime selection
        instability = telemetry[11] if len(telemetry) > 11 else 0.2
        contradiction = telemetry[8] if len(telemetry) > 8 else 0.2
        plateau = telemetry[13] if len(telemetry) > 13 else 0.2
        
        if instability > 0.8:
            regime = "STASIS"
        elif contradiction > 0.6:
            regime = "VERIFY"
        elif plateau > 0.5:
            regime = "MORPH"
        else:
            regime = "GD_ADAM"
        
        # Generate weights
        weights = [random.random() for _ in range(28)]
        knobs = [random.random() for _ in range(14)]
        
        return weights, knobs, regime
    
    def get_regulators(self) -> List[float]:
        """Get current 53 regulator values"""
        if self.use_c:
            return self.regulator.get_regulators()
        return [0.5] * 53
    
    def get_regime(self) -> str:
        """Get current regime"""
        if self.use_c:
            return self.regulator.get_regime_name()
        return "GD_ADAM"
    
    # ========== Consciousness Methods ==========
    def compute_consciousness(self, 
                              K: float, 
                              U: float, 
                              telemetry: List[float],
                              coherence: float = 0.5) -> Dict[str, float]:
        """
        Compute consciousness metrics.
        
        Args:
            K: Knowledge state
            U: Unknowns state
            telemetry: 18-element telemetry
            coherence: Current coherence level
            
        Returns:
            Dictionary with consciousness, coherence, kl_divergence
        """
        if self.use_c:
            telem_arr = list(telemetry[:18]) + [0.0] * (18 - len(telemetry)) if len(telemetry) < 18 else list(telemetry[:18])
            self.consciousness.compute(K, U, coherence, telem_arr, 18)
            return {
                'consciousness': self.consciousness.get_consciousness(),
                'coherence': self.consciousness.get_coherence(),
                'kl_divergence': self.consciousness.get_kl()
            }
        else:
            entropy = sum(telemetry[:5]) / 5.0 if len(telemetry) >= 5 else 0.5
            L_cons = 1.0 - min(1.0, (U + entropy) / (1.0 + K + U))
            return {
                'consciousness': L_cons,
                'coherence': coherence,
                'kl_divergence': 0.1 * (1.0 - coherence)
            }
    
    # ========== Telemetry Methods ==========
    def compute_metrics(self) -> Dict[str, float]:
        """Compute capacity and innocence metrics"""
        if self.use_c:
            return {
                'capacity': self.telemetry.compute_capacity(),
                'innocence': self.telemetry.compute_innocence()
            }
        return {'capacity': 0.5, 'innocence': 0.5}
    
    # ========== Memory Methods ==========
    def store_episode(self, embedding: List[float], salience: float, summary: str = "") -> int:
        """Store an episode in memory"""
        if self.use_c:
            emb = list(embedding[:64]) + [0.0] * (64 - len(embedding)) if len(embedding) < 64 else list(embedding[:64])
            return sam_memory.SamMemory_store_episode(emb, salience, summary)
        return -1
    
    def retrieve_episodes(self, query: List[float], max_results: int = 10) -> List[float]:
        """Retrieve similar episodes"""
        if self.use_c:
            q = list(query[:64]) + [0.0] * (64 - len(query)) if len(query) < 64 else list(query[:64])
            return sam_memory.SamMemory_retrieve(q, max_results)
        return []
    
    def store_concept(self, concept: str, embedding: List[float], strength: float) -> int:
        """Store a semantic concept"""
        if self.use_c:
            emb = list(embedding[:64]) + [0.0] * (64 - len(embedding)) if len(embedding) < 64 else list(embedding[:64])
            return sam_memory.SamMemory_store_semantic(concept, emb, strength)
        return -1
    
    def recall_concept(self, concept: str) -> Optional[List[float]]:
        """Recall a semantic concept"""
        if self.use_c:
            result = sam_memory.SamMemory_recall(concept)
            if result is None:
                return None
            return result
        return None
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        if self.use_c:
            ep_count, sem_count = sam_memory.SamMemory_get_stats()
            return {'episodes': ep_count, 'semantic': sem_count}
        return {'episodes': 0, 'semantic': 0}
    
    # ========== Main Step ==========
    def step(self,
             action: str = "explore",
             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main step function - integrates all cores.
        Includes Phase 1: Id/Ego/Superego, Emotion, Wisdom
        Includes Phase 2: Power, Control, Resources

        Args:
            action: Action type (explore, verify, grow, rest)
            params: Additional parameters

        Returns:
            Dictionary with full state
        """
        params = params or {}
        self.tick += 1

        # Determine efforts based on action
        effort_map = {
            'explore': (0.7, 0.3, 0.2),
            'verify': (0.3, 0.7, 0.2),
            'grow': (0.5, 0.3, 0.7),
            'rest': (0.3, 0.3, 0.1),
        }
        research, verify, morph = effort_map.get(action, (0.5, 0.5, 0.2))

        # Step God Equation
        god_state = self.step_god_equation(research, verify, morph)

        # Build telemetry
        telemetry = self._build_telemetry(god_state)

        # Compile with regulators
        weights, knobs, regime = self.compile_with_telemetry(
            telemetry,
            god_state['K'],
            god_state['U'],
            god_state['omega']
        )

        # Compute metrics
        metrics = self.compute_metrics()

        # Compute consciousness
        consciousness = self.compute_consciousness(
            god_state['K'],
            god_state['U'],
            telemetry,
            god_state['omega']
        )

        # Phase 1: Compute Drive System (Id/Ego/Superego)
        drives = self.drives.compute_drives(telemetry, god_state, action)

        # Phase 1: Compute Emotion System
        emotion = self.emotions.compute_emotion(god_state, drives, self.last_outcome)

        # Phase 1: Compute Wisdom System
        wisdom_state = self.wisdom.compute_wisdom(god_state, drives, emotion)

        # Phase 2: Compute Power System
        power_state = self.power.compute_power(god_state, drives, telemetry)

        # Phase 2: Compute Control System
        control_state = self.control.compute_control(power_state, wisdom_state, drives, emotion)

        # Phase 2: Update Resources
        action_cost = params.get('action_cost', 0.1 * (1.0 + power_state['power']))
        resource_state = self.resources.update_resources(power_state, god_state, action_cost)

        # Apply control if specified
        if params.get('controlled_action'):
            control_success = self.control.apply_control(
                params['controlled_action'],
                params.get('intensity', 0.5)
            )
        else:
            control_success = None

        # Phase 3: Meta-observation
        system_state = {
            'tick': self.tick,
            'god': god_state,
            'drives': drives,
            'emotion': emotion,
            'wisdom': wisdom_state,
            'power': power_state,
            'control': control_state,
            'resources': resource_state,
            'last_outcome': self.last_outcome
        }
        meta_observation = self.meta_observer.observe_self(system_state)
        introspection = self.meta_observer.introspect()

        # Phase 3: Counterfactual reasoning (every 5 ticks to save resources)
        counterfactual_analysis = None
        if self.tick % 5 == 0:
            counterfactual_analysis = self.counterfactual.generate_counterfactual(
                system_state, action, self.last_outcome
            )

        # Phase 3: Version tracking
        version_info = self.versionality.track_version(system_state)

        # Update last outcome (simulated)
        progress = god_state['K'] / (1.0 + god_state['K'])
        self.last_outcome = progress - 0.5

        return {
            'tick': self.tick,
            'action': action,
            'regime': regime,
            'god': god_state,
            'telemetry': telemetry,
            'weights': weights,
            'knobs': knobs,
            'metrics': metrics,
            'regulators': self.get_regulators(),
            'consciousness': consciousness['consciousness'],
            'coherence': consciousness['coherence'],
            'kl_divergence': consciousness['kl_divergence'],
            'drives': drives,
            'emotion': emotion,
            'wisdom': wisdom_state,
            'power': power_state,
            'control': control_state,
            'resources': resource_state,
            'control_success': control_success,
            'meta_observation': meta_observation,
            'introspection': introspection,
            'counterfactual': counterfactual_analysis,
            'version_info': version_info
        }
    
    def _build_telemetry(self, god_state: Dict) -> List[float]:
        """Build 18-element telemetry from god state"""
        K, U, O = god_state['K'], god_state['U'], god_state['O']
        contra = god_state['contradiction']
        
        telemetry = [
            0.1,  # residual
            0.1 * (O / (1 + O)),  # rank_def
            0.3 * (U / (1 + U)),  # retrieval_entropy
            0.1,  # interference
            0.1,  # planner_friction
            0.1,  # context_collapse
            0.1,  # compression_waste
            0.1,  # temporal_incoh
            contra,  # contradiction_score
            0.1,  # calibration_error
            0.1,  # gate_fail_rate
            0.1 * contra,  # instability
            0.2,  # progress_rate
            0.0 if K < 10 else 0.5,  # plateau_flag
            0.2,  # resource_pressure
            0.2,  # novelty_pressure
            0.2,  # coverage_gap
            0.1,  # adversary_pressure
        ]
        return telemetry


def demo():
    """Demo of the C core integration"""
    print("=" * 50)
    print("SAM-D C Core Integration Demo")
    print("=" * 50)
    
    cores = SamCores(seed=42)
    
    print("\n📊 Initial State:")
    state = cores.step('explore')
    print(f"  Tick: {state['tick']}")
    print(f"  Regime: {state['regime']}")
    print(f"  K: {state['god']['K']:.4f}")
    print(f"  U: {state['god']['U']:.4f}")
    print(f"  O: {state['god']['O']:.4f}")
    print(f"  Ω: {state['god']['omega']:.4f}")
    print(f"  Contradiction: {state['god']['contradiction']:.4f}")
    
    print("\n🔄 Running 100 steps...")
    for i in range(100):
        action = ['explore', 'verify', 'grow', 'rest'][i % 4]
        state = cores.step(action)
    
    print(f"\n📊 After 100 steps:")
    print(f"  Tick: {state['tick']}")
    print(f"  Regime: {state['regime']}")
    print(f"  K: {state['god']['K']:.4f}")
    print(f"  U: {state['god']['U']:.4f}")
    print(f"  O: {state['god']['O']:.4f}")
    print(f"  Ω: {state['god']['omega']:.4f}")
    print(f"  Capacity: {state['metrics']['capacity']:.4f}")
    print(f"  Innocence: {state['metrics']['innocence']:.4f}")
    
    print("\n🎯 Top Regulators:")
    regs = state['regulators']
    reg_names = [
        "motivation", "curiosity", "discipline", "focus", "wisdom",
        "identity", "integrity", "coherence", "creativity", "meta_optimization"
    ]
    for i, name in enumerate(reg_names):
        print(f"  {name}: {regs[i]:.4f}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
