#!/usr/bin/env python3
"""
AGI Test Framework
Tests SAM system with training/testing validation split
Tracks AGI-style growth: brittleness, morphogenesis, concept formation
"""

import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not available, plots will be skipped")

sys.path.insert(0, '/Users/samueldasari/Personal/NN_C')

from correct_sam_hub import CorrectSAMHub

class AGITestFramework:
    def __init__(self, test_name="agi_growth_test"):
        self.test_name = test_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"AGI_TEST_RESULTS/{test_name}_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize SAM Hub
        self.hub = CorrectSAMHub()
        
        # Test configuration
        self.config = {
            'epochs': 50,
            'train_split': 0.8,
            'test_split': 0.2,
            'batch_size': 5,
            'morphogenesis_check_interval': 5,
            'validation_interval': 10
        }
        
        # Test data - conversational scenarios that require concept formation
        self.training_scenarios = [
            # Basic concepts
            "What is machine learning?",
            "How does neural network training work?",
            "Explain backpropagation",
            "What is gradient descent?",
            
            # Intermediate concepts requiring synthesis
            "How does attention work in transformers?",
            "Explain the relationship between entropy and information",
            "What is the difference between supervised and unsupervised learning?",
            "How do convolutional networks detect patterns?",
            
            # Advanced concepts requiring new latent dimensions
            "How does meta-learning work?",
            "Explain the bias-variance tradeoff in the context of deep learning",
            "What is the lottery ticket hypothesis?",
            "How does neural architecture search work?",
            
            # Novel concepts that should trigger morphogenesis
            "What would happen if we combined transformers with graph neural networks?",
            "Can we apply information geometry to optimization?",
            "How might compression principles guide architecture design?",
            "What is the relationship between dominant compression and AGI?",
            
            # Complex multi-hop reasoning
            "Given that transformers use attention and GNNs use message passing, could we create a hybrid that uses attention over graph structures?",
            "If dominant compression maximizes J - βH - λC + ηI, how does this relate to the free energy principle?",
            "How does the lottery ticket hypothesis connect to neural tangent kernels?",
            "Can we use morphogenesis to grow new latent dimensions for emergent concepts?"
        ]
        
        # Metrics tracking
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'concepts_born': [],
            'concepts_active': [],
            'concepts_pruned': [],
            'avg_concept_utility': [],
            'brittleness_score': [],
            'morphogenesis_events': [],
            'entropy_score': [],
            'student_teacher_agreement': [],
            'latent_dim': [],
            'verification_rate': []
        }
        
        # Event log
        self.events = []
        
        # Split data
        self._split_data()
        
    def _split_data(self):
        """Split scenarios into train/test"""
        n = len(self.training_scenarios)
        n_train = int(n * self.config['train_split'])
        
        # Shuffle
        import random
        random.seed(42)
        shuffled = self.training_scenarios.copy()
        random.shuffle(shuffled)
        
        self.train_data = shuffled[:n_train]
        self.test_data = shuffled[n_train:]
        
        print(f"📊 Data split: {len(self.train_data)} train, {len(self.test_data)} test")
        
    def run_epoch(self, epoch_num):
        """Run one training epoch"""
        print(f"\n{'='*60}")
        print(f"🔄 EPOCH {epoch_num + 1}/{self.config['epochs']}")
        print(f"{'='*60}")
        
        epoch_metrics = {
            'train_errors': [],
            'verified_responses': 0,
            'total_responses': 0,
            'morphogenesis_triggered': False,
            'new_concepts': 0,
            'concepts_pruned': 0
        }
        
        # Training phase
        print(f"\n📚 Training Phase ({len(self.train_data)} scenarios)")
        for i, scenario in enumerate(self.train_data):
            print(f"  [{i+1}/{len(self.train_data)}] Processing: {scenario[:50]}...")
            
            # Process with search pipeline
            response, is_verified, confidence = self.hub.process_with_search_pipeline(
                scenario, 
                'sam_conversation'
            )
            
            epoch_metrics['total_responses'] += 1
            if is_verified:
                epoch_metrics['verified_responses'] += 1
            
            # Simulate error (would come from actual training in full implementation)
            simulated_error = 0.3 - (0.2 * confidence) + (0.1 * (1 - is_verified))
            epoch_metrics['train_errors'].append(simulated_error)
            
            # Check for morphogenesis
            if i % self.config['morphogenesis_check_interval'] == 0:
                if self.hub.check_morphogenesis_trigger(simulated_error):
                    concept_id = self.hub.birth_concept(
                        concept_name=f"emergent_concept_{epoch_num}_{i}",
                        context_trigger=scenario[:30]
                    )
                    epoch_metrics['morphogenesis_triggered'] = True
                    epoch_metrics['new_concepts'] += 1
                    
                    self.events.append({
                        'epoch': epoch_num,
                        'step': i,
                        'type': 'concept_birth',
                        'concept_id': concept_id,
                        'trigger': scenario[:50],
                        'error': simulated_error
                    })
            
            # Update concept utilities
            self.hub.update_concept_utilities(simulated_error)
            
            time.sleep(0.1)  # Brief pause
        
        # Testing phase
        print(f"\n🧪 Testing Phase ({len(self.test_data)} scenarios)")
        test_errors = []
        test_verified = 0
        
        for i, scenario in enumerate(self.test_data):
            response, is_verified, confidence = self.hub.process_with_search_pipeline(
                scenario,
                'sam_conversation'
            )
            
            simulated_error = 0.3 - (0.2 * confidence) + (0.1 * (1 - is_verified))
            test_errors.append(simulated_error)
            if is_verified:
                test_verified += 1
        
        # Calculate metrics
        train_loss = sum(epoch_metrics['train_errors']) / len(epoch_metrics['train_errors'])
        test_loss = sum(test_errors) / len(test_errors) if test_errors else 0
        train_acc = epoch_metrics['verified_responses'] / epoch_metrics['total_responses']
        test_acc = test_verified / len(self.test_data) if self.test_data else 0
        
        # Get morphogenesis summary
        mg_summary = self.hub.get_morphogenesis_summary()
        
        # Record metrics
        self.metrics['epoch'].append(epoch_num)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['train_accuracy'].append(train_acc)
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['concepts_born'].append(mg_summary['total_concepts'])
        self.metrics['concepts_active'].append(mg_summary['active_concepts'])
        self.metrics['concepts_pruned'].append(mg_summary['dead_concepts'])
        self.metrics['avg_concept_utility'].append(mg_summary['avg_utility'])
        self.metrics['brittleness_score'].append(self._calculate_brittleness())
        self.metrics['morphogenesis_events'].append(1 if epoch_metrics['morphogenesis_triggered'] else 0)
        self.metrics['latent_dim'].append(mg_summary['total_concepts'])
        self.metrics['verification_rate'].append(train_acc)
        
        # Validation metrics
        val_metrics = self.hub.get_validation_metrics()
        self.metrics['entropy_score'].append(val_metrics.get('entropy_score', 0) or 0)
        self.metrics['student_teacher_agreement'].append(val_metrics.get('agreement', 0) or 0)
        
        # Print summary
        print(f"\n📈 Epoch {epoch_num + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"   Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
        print(f"   Concepts: {mg_summary['active_concepts']} active, {mg_summary['dead_concepts']} pruned")
        print(f"   Brittleness: {self._calculate_brittleness():.4f}")
        print(f"   Morphogenesis: {epoch_metrics['new_concepts']} new concepts")
        
        return epoch_metrics
    
    def _calculate_brittleness(self):
        """Calculate system brittleness score"""
        # High brittleness = high error + high variance + stuck optimization
        if len(self.hub.error_history) < 10:
            return 0.5
        
        recent_errors = self.hub.error_history[-10:]
        mean_error = sum(recent_errors) / len(recent_errors)
        variance = sum((e - mean_error)**2 for e in recent_errors) / len(recent_errors)
        
        # Trend (negative = improving, positive = stuck)
        trend = self.hub._compute_error_trend(recent_errors)
        
        # Brittleness increases with: high error, high variance, positive trend
        brittleness = mean_error + variance + max(0, trend)
        return min(1.0, brittleness)
    
    def run_full_test(self):
        """Run complete test over all epochs"""
        print("\n" + "="*70)
        print("🧠 AGI GROWTH TEST FRAMEWORK")
        print("="*70)
        print(f"Test: {self.test_name}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Train/Test Split: {self.config['train_split']}/{self.config['test_split']}")
        print(f"Results Directory: {self.results_dir}")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.run_epoch(epoch)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ TEST COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*70}")
        
        # Generate final report
        self._generate_report()
        
        return self.metrics
    
    def _save_checkpoint(self, epoch):
        """Save checkpoint with current state"""
        checkpoint = {
            'epoch': epoch,
            'metrics': {k: v[:] for k, v in self.metrics.items()},  # Copy lists
            'concepts': list(self.hub.concept_registry.values()),
            'events': self.events,
            'config': self.config
        }
        
        path = f"{self.results_dir}/checkpoint_epoch_{epoch}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"💾 Checkpoint saved: {path}")
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Generating Final Report...")
        
        # Save final metrics
        metrics_path = f"{self.results_dir}/final_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save events
        events_path = f"{self.results_dir}/events.json"
        with open(events_path, 'w') as f:
            json.dump(self.events, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
        # Generate text summary
        self._generate_text_summary()
        
        print(f"📁 Results saved to: {self.results_dir}")
    
    def _generate_plots(self):
        """Generate visualization plots"""
        if not HAS_MATPLOTLIB:
            print("📈 Skipping plots (matplotlib not available)")
            return
            
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('AGI Growth Test Results', fontsize=16)
            
            epochs = self.metrics['epoch']
            
            # Plot 1: Loss curves
            ax = axes[0, 0]
            ax.plot(epochs, self.metrics['train_loss'], label='Train Loss', marker='o')
            ax.plot(epochs, self.metrics['test_loss'], label='Test Loss', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True)
            
            # Plot 2: Accuracy curves
            ax = axes[0, 1]
            ax.plot(epochs, self.metrics['train_accuracy'], label='Train Accuracy', marker='o')
            ax.plot(epochs, self.metrics['test_accuracy'], label='Test Accuracy', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Verification Rate')
            ax.legend()
            ax.grid(True)
            
            # Plot 3: Concept growth
            ax = axes[0, 2]
            ax.plot(epochs, self.metrics['concepts_born'], label='Total Born', marker='o')
            ax.plot(epochs, self.metrics['concepts_active'], label='Active', marker='s')
            ax.plot(epochs, self.metrics['concepts_pruned'], label='Pruned', marker='^')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Concept Count')
            ax.set_title('Concept Morphogenesis')
            ax.legend()
            ax.grid(True)
            
            # Plot 4: Brittleness
            ax = axes[1, 0]
            ax.plot(epochs, self.metrics['brittleness_score'], color='red', marker='o')
            ax.axhline(y=0.15, color='orange', linestyle='--', label='Morphogenesis Threshold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Brittleness Score')
            ax.set_title('System Brittleness')
            ax.legend()
            ax.grid(True)
            
            # Plot 5: Concept Utility
            ax = axes[1, 1]
            ax.plot(epochs, self.metrics['avg_concept_utility'], color='green', marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Utility')
            ax.set_title('Concept Quality')
            ax.grid(True)
            
            # Plot 6: Latent Dimension Growth
            ax = axes[1, 2]
            ax.plot(epochs, self.metrics['latent_dim'], color='purple', marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Latent Dimensions')
            ax.set_title('Network Capacity Growth')
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/growth_plots.png", dpi=150)
            print(f"📈 Plots saved: {self.results_dir}/growth_plots.png")
            
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
    
    def _generate_text_summary(self):
        """Generate text summary of test results"""
        summary = []
        summary.append("="*70)
        summary.append("AGI GROWTH TEST - FINAL SUMMARY")
        summary.append("="*70)
        summary.append(f"Test Name: {self.test_name}")
        summary.append(f"Timestamp: {self.timestamp}")
        summary.append(f"Epochs: {self.config['epochs']}")
        summary.append("")
        
        # Final metrics
        summary.append("📊 FINAL METRICS:")
        summary.append(f"  Train Loss: {self.metrics['train_loss'][-1]:.4f}")
        summary.append(f"  Test Loss: {self.metrics['test_loss'][-1]:.4f}")
        summary.append(f"  Train Accuracy: {self.metrics['train_accuracy'][-1]:.2%}")
        summary.append(f"  Test Accuracy: {self.metrics['test_accuracy'][-1]:.2%}")
        summary.append(f"  Total Concepts Born: {self.metrics['concepts_born'][-1]}")
        summary.append(f"  Active Concepts: {self.metrics['concepts_active'][-1]}")
        summary.append(f"  Pruned Concepts: {self.metrics['concepts_pruned'][-1]}")
        summary.append(f"  Final Brittleness: {self.metrics['brittleness_score'][-1]:.4f}")
        summary.append("")
        
        # Growth analysis
        summary.append("📈 GROWTH ANALYSIS:")
        concept_growth = self.metrics['concepts_born'][-1] - self.metrics['concepts_born'][0]
        loss_reduction = self.metrics['train_loss'][0] - self.metrics['train_loss'][-1]
        acc_improvement = self.metrics['train_accuracy'][-1] - self.metrics['train_accuracy'][0]
        
        summary.append(f"  Concept Growth: {concept_growth} new concepts")
        summary.append(f"  Loss Reduction: {loss_reduction:.4f}")
        summary.append(f"  Accuracy Improvement: {acc_improvement:.2%}")
        summary.append("")
        
        # Morphogenesis events
        morph_events = sum(self.metrics['morphogenesis_events'])
        summary.append(f"🧬 MORPHOGENESIS EVENTS: {morph_events}")
        if morph_events > 0:
            summary.append("  Concept birth events occurred during training")
            summary.append("  System expanded latent dimensions dynamically")
        summary.append("")
        
        # Key observations
        summary.append("🔍 KEY OBSERVATIONS:")
        if loss_reduction > 0.1:
            summary.append("  ✅ Significant learning occurred (loss reduced)")
        if concept_growth > 5:
            summary.append("  ✅ Active concept formation (morphogenesis working)")
        if self.metrics['brittleness_score'][-1] < 0.2:
            summary.append("  ✅ Low final brittleness (stable system)")
        if acc_improvement > 0.2:
            summary.append("  ✅ Strong verification improvement")
        summary.append("")
        
        # AGI indicators
        summary.append("🧠 AGI-STYLE GROWTH INDICATORS:")
        has_self_organization = concept_growth > 3
        has_adaptation = loss_reduction > 0.05
        has_verification = self.metrics['train_accuracy'][-1] > 0.7
        
        summary.append(f"  Self-Organization: {'✅' if has_self_organization else '❌'} (concept formation)")
        summary.append(f"  Adaptive Learning: {'✅' if has_adaptation else '❌'} (error reduction)")
        summary.append(f"  Knowledge Verification: {'✅' if has_verification else '❌'} (fact-checking)")
        summary.append("")
        
        summary.append("="*70)
        
        # Save summary
        summary_text = "\n".join(summary)
        with open(f"{self.results_dir}/summary.txt", 'w') as f:
            f.write(summary_text)
        
        print(summary_text)

if __name__ == "__main__":
    # Run the test
    framework = AGITestFramework()
    metrics = framework.run_full_test()
    
    print("\n✅ Test framework execution complete!")
    print(f"Results available in: {framework.results_dir}")
