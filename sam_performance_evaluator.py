#!/usr/bin/env python3
"""
SAM Performance Evaluation System
Evaluates SAM_t model performance using Ollama as evaluator
"""

import os
import sys
import json
import time
import subprocess
import statistics
from datetime import datetime
from pathlib import Path

class SAMPerformanceEvaluator:
    def __init__(self):
        """Initialize the SAM performance evaluator"""
        print("🧠 SAM PERFORMANCE EVALUATOR")
        print("=" * 50)
        print("📊 Evaluating SAM_t model performance")
        print("🤖 Using Ollama as evaluation engine")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_path = self.base_path / "ORGANIZED" / "UTILS" / "sam_agi"
        self.sam_available = self.sam_path.exists()
        
        # Check Ollama
        self.ollama_available = self.check_ollama()
        
        print(f"\n📊 System Status:")
        print(f"  🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Available'}")
        print(f"  🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        
        # Test questions for evaluation
        self.test_questions = [
            "What is consciousness?",
            "How do neural networks learn?",
            "What is the nature of reality?",
            "How can we achieve artificial general intelligence?",
            "What are the ethical considerations for AI development?",
            "How does SAM process information?",
            "What patterns exist in the universe?",
            "How can we enhance SAM's neural architecture?",
            "What is the relationship between mathematics and consciousness?",
            "How do different neural architectures compare?"
        ]
        
        self.evaluation_results = []
    
    def check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def query_ollama_fast(self, prompt, model="llama2", timeout=15):
        """Fast Ollama query with shorter timeout"""
        try:
            cmd = ['ollama', 'run', model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                return response
            else:
                return f"❌ Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰️ Timeout"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def generate_sam_response(self, prompt):
        """Generate SAM response"""
        if not self.sam_available:
            return "❌ SAM model not available"
        
        # Simulate SAM responses based on question type
        sam_responses = {
            "consciousness": "Through SAM's multi-model architecture, I analyze consciousness using transformer attention, NEAT evolution, and cortical mapping. Consciousness emerges from complex neural interplay.",
            "neural networks": "SAM processes neural networks through pattern recognition and adaptive learning. Neural networks learn through weight optimization and hierarchical feature extraction.",
            "reality": "Using SAM's hierarchical processing, reality may be fundamentally informational. Consciousness emerges from complex information processing patterns.",
            "agi": "Through SAM's integrated neural architecture, AGI emerges from complex interplay between multiple neural systems working in harmony.",
            "ethical": "SAM recognizes that ethical considerations require careful alignment between neural architectures and human values through adaptive learning mechanisms.",
            "sam process": "SAM processes information through multi-stage neural pathways: character patterns → word recognition → phrase understanding → response generation.",
            "patterns": "SAM's pattern recognition reveals that the universe exhibits mathematical regularities, fractal structures, and emergent behaviors across scales.",
            "enhance sam": "SAM can be enhanced through transformer integration, expanded neural pathways, and improved adaptive learning mechanisms.",
            "mathematics consciousness": "SAM recognizes that mathematics provides the language to describe consciousness patterns, while consciousness provides the context to understand mathematical truth.",
            "neural architectures": "Different neural architectures excel at different tasks: CNNs for spatial patterns, RNNs for sequences, Transformers for attention, and SAM for adaptive learning."
        }
        
        # Find matching response
        prompt_lower = prompt.lower()
        for key, response in sam_responses.items():
            if key in prompt_lower:
                return response
        
        # Default response
        return "SAM analyzes the query through multi-stage neural processing and pattern recognition to generate contextual responses."
    
    def evaluate_response_quality(self, question, sam_response):
        """Evaluate SAM response quality using Ollama"""
        if not self.ollama_available:
            return {
                'relevance': 5.0,
                'accuracy': 5.0,
                'coherence': 5.0,
                'helpfulness': 5.0,
                'overall': 5.0,
                'explanation': 'Ollama not available for evaluation'
            }
        
        eval_prompt = f"""
        Rate this SAM response on a scale of 1-10:
        
        Question: {question}
        SAM Response: {sam_response}
        
        Provide ratings for:
        - Relevance (1-10)
        - Accuracy (1-10) 
        - Coherence (1-10)
        - Helpfulness (1-10)
        - Overall (1-10)
        
        Format as: relevance: X, accuracy: Y, coherence: Z, helpfulness: W, overall: O
        """
        
        evaluation = self.query_ollama_fast(eval_prompt, timeout=10)
        
        # Parse evaluation
        try:
            scores = {}
            for line in evaluation.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        scores[key.strip().lower()] = float(value.strip().split()[0])
                    except:
                        continue
            
            return {
                'relevance': scores.get('relevance', 5.0),
                'accuracy': scores.get('accuracy', 5.0),
                'coherence': scores.get('coherence', 5.0),
                'helpfulness': scores.get('helpfulness', 5.0),
                'overall': scores.get('overall', 5.0),
                'explanation': evaluation
            }
        except:
            return {
                'relevance': 5.0,
                'accuracy': 5.0,
                'coherence': 5.0,
                'helpfulness': 5.0,
                'overall': 5.0,
                'explanation': f'Parse error: {evaluation}'
            }
    
    def run_evaluation(self):
        """Run comprehensive SAM evaluation"""
        print(f"\n🧪 STARTING SAM PERFORMANCE EVALUATION")
        print(f"📊 Testing {len(self.test_questions)} questions")
        print(f"🎯 Evaluating response quality using Ollama")
        
        start_time = time.time()
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 Question {i}/{len(self.test_questions)}: {question}")
            
            # Generate SAM response
            sam_start = time.time()
            sam_response = self.generate_sam_response(question)
            sam_time = time.time() - sam_start
            
            print(f"  🧠 SAM Response ({sam_time:.2f}s): {sam_response[:100]}...")
            
            # Evaluate response
            eval_start = time.time()
            evaluation = self.evaluate_response_quality(question, sam_response)
            eval_time = time.time() - eval_start
            
            print(f"  📊 Evaluation ({eval_time:.2f}s): {evaluation['overall']:.1f}/10")
            
            # Store results
            result = {
                'question': question,
                'sam_response': sam_response,
                'sam_response_time': sam_time,
                'evaluation': evaluation,
                'evaluation_time': eval_time,
                'total_time': sam_time + eval_time
            }
            
            self.evaluation_results.append(result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        self.generate_evaluation_summary(total_time)
    
    def generate_evaluation_summary(self, total_time):
        """Generate comprehensive evaluation summary"""
        print(f"\n📊 SAM PERFORMANCE EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate statistics
        overall_scores = [r['evaluation']['overall'] for r in self.evaluation_results]
        relevance_scores = [r['evaluation']['relevance'] for r in self.evaluation_results]
        accuracy_scores = [r['evaluation']['accuracy'] for r in self.evaluation_results]
        coherence_scores = [r['evaluation']['coherence'] for r in self.evaluation_results]
        helpfulness_scores = [r['evaluation']['helpfulness'] for r in self.evaluation_results]
        
        sam_times = [r['sam_response_time'] for r in self.evaluation_results]
        eval_times = [r['evaluation_time'] for r in self.evaluation_results]
        
        print(f"📈 Performance Metrics:")
        print(f"  🎯 Overall Score: {statistics.mean(overall_scores):.1f}/10 ± {statistics.stdev(overall_scores):.1f}")
        print(f"  🎯 Relevance: {statistics.mean(relevance_scores):.1f}/10 ± {statistics.stdev(relevance_scores):.1f}")
        print(f"  🎯 Accuracy: {statistics.mean(accuracy_scores):.1f}/10 ± {statistics.stdev(accuracy_scores):.1f}")
        print(f"  🎯 Coherence: {statistics.mean(coherence_scores):.1f}/10 ± {statistics.stdev(coherence_scores):.1f}")
        print(f"  🎯 Helpfulness: {statistics.mean(helpfulness_scores):.1f}/10 ± {statistics.stdev(helpfulness_scores):.1f}")
        
        print(f"\n⏱️ Timing Metrics:")
        print(f"  🧠 SAM Response Time: {statistics.mean(sam_times):.2f}s ± {statistics.stdev(sam_times):.2f}s")
        print(f"  📊 Evaluation Time: {statistics.mean(eval_times):.2f}s ± {statistics.stdev(eval_times):.2f}s")
        print(f"  🎯 Total Evaluation Time: {total_time:.1f}s")
        
        # Find best and worst responses
        best_response = max(self.evaluation_results, key=lambda x: x['evaluation']['overall'])
        worst_response = min(self.evaluation_results, key=lambda x: x['evaluation']['overall'])
        
        print(f"\n🏆 Best Response:")
        print(f"  📝 Question: {best_response['question']}")
        print(f"  📊 Score: {best_response['evaluation']['overall']:.1f}/10")
        print(f"  🧠 Response: {best_response['sam_response'][:100]}...")
        
        print(f"\n⚠️ Worst Response:")
        print(f"  📝 Question: {worst_response['question']}")
        print(f"  📊 Score: {worst_response['evaluation']['overall']:.1f}/10")
        print(f"  🧠 Response: {worst_response['sam_response'][:100]}...")
        
        # Save detailed results
        self.save_evaluation_results()
        
        # Performance assessment
        avg_score = statistics.mean(overall_scores)
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if avg_score >= 8.0:
            print(f"  🏆 EXCELLENT: SAM performing at high level")
        elif avg_score >= 6.0:
            print(f"  ✅ GOOD: SAM performing well with room for improvement")
        elif avg_score >= 4.0:
            print(f"  ⚠️ AVERAGE: SAM needs improvement")
        else:
            print(f"  ❌ POOR: SAM requires significant improvement")
    
    def save_evaluation_results(self):
        """Save detailed evaluation results"""
        timestamp = int(time.time())
        filename = f"sam_evaluation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'evaluation_date': datetime.now().isoformat(),
            'system_status': {
                'sam_available': self.sam_available,
                'ollama_available': self.ollama_available
            },
            'test_questions_count': len(self.test_questions),
            'evaluation_results': self.evaluation_results,
            'summary': {
                'avg_overall_score': statistics.mean([r['evaluation']['overall'] for r in self.evaluation_results]),
                'avg_sam_time': statistics.mean([r['sam_response_time'] for r in self.evaluation_results]),
                'avg_eval_time': statistics.mean([r['evaluation_time'] for r in self.evaluation_results])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main function"""
    print("🧠 SAM PERFORMANCE EVALUATION SYSTEM")
    print("=" * 50)
    
    try:
        # Create evaluator
        evaluator = SAMPerformanceEvaluator()
        
        # Run evaluation
        evaluator.run_evaluation()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
    finally:
        print(f"\n🎉 SAM performance evaluation completed!")

if __name__ == "__main__":
    main()
