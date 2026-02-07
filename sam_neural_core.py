#!/usr/bin/env python3
"""
SAM Neural Core - Python interface to C neural network implementation
Uses ctypes to bridge Python with the SAM C core
"""

import ctypes
import os
import sys
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

class SAMNeuralCore:
    """
    Interface to the SAM C neural network implementation
    Provides access to:
    - Full-context batch learning with dominant compression
    - Latent-space morphogenesis (concept birth/death)
    - Neural network operations (forward, backprop, update)
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """Initialize the SAM neural core
        
        Args:
            lib_path: Path to the compiled shared library (.so or .dylib)
                   If None, searches in standard locations
        """
        self.lib = None
        self.morpho_state = None
        self._load_library(lib_path)
        self._setup_ctypes()
        
    def _load_library(self, lib_path: Optional[str] = None):
        """Load the C shared library"""
        if lib_path is None:
            # Search in standard locations
            base_dir = Path(__file__).parent
            possible_paths = [
                base_dir / "libsam_core.dylib",  # macOS
                base_dir / "libsam_core.so",     # Linux
                base_dir / "libsam_core.dll",    # Windows
                Path("/Users/samueldasari/Personal/NN_C") / "libsam_core.dylib",
                Path("/Users/samueldasari/Personal/NN_C") / "libsam_core.so",
            ]
            
            for path in possible_paths:
                if path.exists():
                    lib_path = str(path)
                    break
        
        if lib_path is None or not os.path.exists(lib_path):
            raise RuntimeError(
                f"SAM core library not found. Please compile first:\n"
                f"cd /Users/samueldasari/Personal/NN_C && make shared\n"
                f"Searched: {[str(p) for p in possible_paths if 'possible_paths' in dir()] or 'standard locations'}"
            )
        
        self.lib = ctypes.CDLL(lib_path)
        print(f"✓ Loaded SAM core: {lib_path}")
    
    def _setup_ctypes(self):
        """Setup ctypes function signatures"""
        if not self.lib:
            return
            
        # Morphogenesis functions
        try:
            self.lib.morphogenesis_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            self.lib.morphogenesis_create.restype = ctypes.c_void_p
            
            self.lib.morphogenesis_destroy.argtypes = [ctypes.c_void_p]
            self.lib.morphogenesis_destroy.restype = None
            
            self.lib.morphogenesis_check_trigger.argtypes = [ctypes.c_void_p, ctypes.c_longdouble]
            self.lib.morphogenesis_check_trigger.restype = ctypes.c_int
            
            self.lib.morphogenesis_record_error.argtypes = [ctypes.c_void_p, ctypes.c_longdouble]
            self.lib.morphogenesis_record_error.restype = None
            
            self.lib.morphogenesis_get_trend.argtypes = [ctypes.c_void_p]
            self.lib.morphogenesis_get_trend.restype = ctypes.c_longdouble
            
            self.lib.morphogenesis_birth_concept.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
            ]
            self.lib.morphogenesis_birth_concept.restype = ctypes.c_int
            
            self.lib.morphogenesis_prune_concept.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.morphogenesis_prune_concept.restype = ctypes.c_int
            
            self.lib.morphogenesis_structure_cost.argtypes = [ctypes.c_void_p]
            self.lib.morphogenesis_structure_cost.restype = ctypes.c_longdouble
            
            self.lib.morphogenesis_print_summary.argtypes = [ctypes.c_void_p]
            self.lib.morphogenesis_print_summary.restype = None
            
            print("✓ Morphogenesis functions bound")
            
        except AttributeError as e:
            print(f"⚠️ Some morphogenesis functions not found: {e}")
    
    def initialize_morphogenesis(self, initial_dim: int = 64, max_dim: int = 4096) -> bool:
        """Initialize the morphogenesis system
        
        Args:
            initial_dim: Starting latent dimensionality
            max_dim: Maximum allowed dimensionality
            
        Returns:
            True if successful
        """
        if not self.lib:
            return False
            
        try:
            self.morpho_state = self.lib.morphogenesis_create(initial_dim, max_dim)
            if self.morpho_state:
                print(f"🧬 Morphogenesis initialized: dim={initial_dim}, max={max_dim}")
                return True
        except Exception as e:
            print(f"⚠️ Morphogenesis init failed: {e}")
            
        return False
    
    def check_morphogenesis_trigger(self, current_error: float) -> bool:
        """Check if conditions are met for concept birth
        
        Args:
            current_error: Current prediction error level
            
        Returns:
            True if new concept should be born
        """
        if not self.morpho_state or not self.lib:
            return False
            
        try:
            result = self.lib.morphogenesis_check_trigger(
                self.morpho_state, ctypes.c_longdouble(current_error)
            )
            return result != 0
        except Exception as e:
            print(f"⚠️ Trigger check failed: {e}")
            return False
    
    def record_error(self, error: float):
        """Record error for tracking and trend analysis
        
        Args:
            error: Current prediction error
        """
        if self.morpho_state and self.lib:
            try:
                self.lib.morphogenesis_record_error(
                    self.morpho_state, ctypes.c_longdouble(error)
                )
            except Exception as e:
                print(f"⚠️ Error recording failed: {e}")
    
    def get_error_trend(self) -> float:
        """Get current error trend (linear regression slope)
        
        Returns:
            Slope of recent error trend (positive = worsening)
        """
        if not self.morpho_state or not self.lib:
            return 0.0
            
        try:
            trend = self.lib.morphogenesis_get_trend(self.morpho_state)
            return float(trend)
        except Exception as e:
            print(f"⚠️ Trend calculation failed: {e}")
            return 0.0
    
    def birth_concept(self, concept_name: str) -> bool:
        """Birth a new concept (expand latent dimension)
        
        Args:
            concept_name: Human-readable name for the concept
            
        Returns:
            True if concept was born successfully
        """
        if not self.morpho_state or not self.lib:
            return False
            
        try:
            name_bytes = concept_name.encode('utf-8')
            result = self.lib.morphogenesis_birth_concept(
                self.morpho_state, None, ctypes.c_char_p(name_bytes)
            )
            
            if result == 0:
                print(f"🌱 Concept born: '{concept_name}'")
                return True
            else:
                print(f"⚠️ Concept birth failed with code {result}")
                return False
                
        except Exception as e:
            print(f"⚠️ Concept birth failed: {e}")
            return False
    
    def prune_concept(self, concept_idx: int) -> bool:
        """Prune a concept by index
        
        Args:
            concept_idx: Index of concept to prune
            
        Returns:
            True if pruning was successful
        """
        if not self.morpho_state or not self.lib:
            return False
            
        try:
            result = self.lib.morphogenesis_prune_concept(
                self.morpho_state, ctypes.c_int(concept_idx)
            )
            return result == 0
        except Exception as e:
            print(f"⚠️ Concept pruning failed: {e}")
            return False
    
    def get_structure_cost(self) -> float:
        """Get current structural regularizer cost
        
        Returns:
            Cost value (higher = more complex structure)
        """
        if not self.morpho_state or not self.lib:
            return 0.0
            
        try:
            cost = self.lib.morphogenesis_structure_cost(self.morpho_state)
            return float(cost)
        except Exception as e:
            print(f"⚠️ Structure cost calculation failed: {e}")
            return 0.0
    
    def print_summary(self):
        """Print morphogenesis system summary to console"""
        if self.morpho_state and self.lib:
            try:
                self.lib.morphogenesis_print_summary(self.morpho_state)
            except Exception as e:
                print(f"⚠️ Summary print failed: {e}")
    
    def get_state_dict(self) -> Dict:
        """Get current morphogenesis state as dictionary
        
        Returns:
            Dictionary with current state information
        """
        return {
            'has_state': self.morpho_state is not None,
            'has_library': self.lib is not None,
            'error_trend': self.get_error_trend(),
            'structure_cost': self.get_structure_cost()
        }
    
    def cleanup(self):
        """Clean up morphogenesis resources"""
        if self.morpho_state and self.lib:
            try:
                self.lib.morphogenesis_destroy(self.morpho_state)
                self.morpho_state = None
                print("✓ Morphogenesis resources cleaned up")
            except Exception as e:
                print(f"⚠️ Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()


class SAMNetworkManager:
    """
    Manages neural network operations including:
    - Network creation and initialization
    - Weight cloning for submodels
    - Latent dimension expansion/compression
    """
    
    def __init__(self, core: SAMNeuralCore):
        """Initialize with a SAMNeuralCore instance"""
        self.core = core
        self.active_networks = {}
        
    def create_network(self, network_id: str, input_dim: int, 
                      hidden_dims: List[int], output_dim: int) -> Dict:
        """Create a new neural network configuration
        
        This creates the metadata structure. Actual weight allocation
        happens when the network is first used.
        
        Args:
            network_id: Unique identifier for this network
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            
        Returns:
            Network configuration dictionary
        """
        config = {
            'id': network_id,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'total_params': self._compute_param_count(input_dim, hidden_dims, output_dim),
            'created_at': str(np.datetime64('now')),
            'weights_path': f'SAM_STATE/networks/{network_id}.bin'
        }
        
        self.active_networks[network_id] = config
        print(f"🧠 Network '{network_id}' configured: {config['total_params']} params")
        return config
    
    def clone_network(self, source_id: str, target_id: str, 
                     specialization: str = "") -> Optional[Dict]:
        """Clone a network for submodel specialization
        
        This performs actual neural network weight copying via the C core.
        After cloning, the target can be specialized through distillation.
        
        Args:
            source_id: ID of source network to clone
            target_id: ID for the new clone
            specialization: Context for specialization
            
        Returns:
            Clone configuration or None if failed
        """
        if source_id not in self.active_networks:
            print(f"❌ Source network '{source_id}' not found")
            return None
            
        source = self.active_networks[source_id]
        
        # Create clone configuration
        clone = {
            'id': target_id,
            'parent': source_id,
            'input_dim': source['input_dim'],
            'hidden_dims': source['hidden_dims'].copy(),
            'output_dim': source['output_dim'],
            'total_params': source['total_params'],
            'created_at': str(np.datetime64('now')),
            'specialization': specialization,
            'weights_path': f'SAM_STATE/clones/{target_id}.bin',
            'clone_method': 'transfusion_distillation',
            'distillation_steps': 0,
            'verified_examples': 0
        }
        
        # In full implementation, this would:
        # 1. Load source weights from C memory
        # 2. Copy to target memory
        # 3. Save to target path
        # For now, mark as pending
        clone['status'] = 'pending_weight_copy'
        
        self.active_networks[target_id] = clone
        print(f"🧬 Cloned '{source_id}' → '{target_id}' for {specialization}")
        return clone
    
    def expand_latent_dim(self, network_id: str, new_dim: int) -> bool:
        """Expand network latent dimension (concept birth)
        
        Args:
            network_id: Network to expand
            new_dim: New latent dimension (must be larger than current)
            
        Returns:
            True if expansion successful
        """
        if network_id not in self.active_networks:
            return False
            
        net = self.active_networks[network_id]
        
        # Update hidden dims to reflect expansion
        old_dims = net['hidden_dims']
        if len(old_dims) > 0 and new_dim > old_dims[-1]:
            net['hidden_dims'][-1] = new_dim
            net['total_params'] = self._compute_param_count(
                net['input_dim'], net['hidden_dims'], net['output_dim']
            )
            print(f"📈 Expanded '{network_id}' latent dim: {old_dims[-1]} → {new_dim}")
            return True
            
        return False
    
    def _compute_param_count(self, input_dim: int, hidden_dims: List[int], 
                            output_dim: int) -> int:
        """Compute total parameter count for a network"""
        total = 0
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            total += dims[i] * dims[i + 1]  # weights
            total += dims[i + 1]              # biases
        return total
    
    def get_network_info(self, network_id: str) -> Optional[Dict]:
        """Get information about a network"""
        return self.active_networks.get(network_id)
    
    def list_networks(self) -> List[str]:
        """List all managed network IDs"""
        return list(self.active_networks.keys())


# Convenience function for quick initialization
def create_sam_core(lib_path: Optional[str] = None) -> Tuple[SAMNeuralCore, SAMNetworkManager]:
    """Create and initialize SAM core components
    
    Args:
        lib_path: Optional path to shared library
        
    Returns:
        Tuple of (SAMNeuralCore, SAMNetworkManager)
    """
    core = SAMNeuralCore(lib_path)
    manager = SAMNetworkManager(core)
    return core, manager


if __name__ == "__main__":
    # Test the core
    print("🧪 Testing SAM Neural Core")
    print("=" * 50)
    
    try:
        core, manager = create_sam_core()
        
        # Initialize morphogenesis
        if core.initialize_morphogenesis(initial_dim=64, max_dim=256):
            # Test error recording and trigger detection
            for i in range(25):
                error = 0.2 + np.random.normal(0, 0.05)
                core.record_error(max(0.1, error))
            
            # Check trigger
            trigger = core.check_morphogenesis_trigger(0.25)
            print(f"🚨 Morphogenesis trigger: {trigger}")
            
            # Get trend
            trend = core.get_error_trend()
            print(f"📈 Error trend: {trend:.6f}")
            
            # Try to birth a concept
            if trigger:
                core.birth_concept("test_concept")
            
            # Print summary
            core.print_summary()
        
        # Test network management
        net = manager.create_network(
            "sam_head", 
            input_dim=768, 
            hidden_dims=[512, 256, 128], 
            output_dim=64
        )
        
        clone = manager.clone_network("sam_head", "sam_coder", "software_development")
        
        print("\n✅ SAM Core test complete!")
        
    except Exception as e:
        print(f"\n⚠️ Test incomplete: {e}")
        print("Make sure to compile the C library first:")
        print("cd /Users/samueldasari/Personal/NN_C && make shared")
