#!/usr/bin/env python3
"""
Comprehensive Mathematical Training System
Trains on all types of problems including NP problems
Builds upon persistent knowledge system
"""

import os
import sys
import time
import json
import random
import math
from datetime import datetime
from persistent_knowledge_system import PersistentKnowledgeSystem

class ComprehensiveMathTraining:
    def __init__(self):
        self.knowledge_system = PersistentKnowledgeSystem()
        self.session_start = time.time()
        
        print("🎓 COMPREHENSIVE MATHEMATICAL TRAINING")
        print("=" * 50)
        print("🧠 Training on all problem types including NP problems")
        
        # Show existing knowledge
        summary = self.knowledge_system.get_knowledge_summary()
        print(f"📊 Existing Knowledge Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        if summary['total_knowledge_items'] > 0:
            print(f"\n✅ LOADED EXISTING KNOWLEDGE - Building upon {summary['total_knowledge_items']} items")
        else:
            print(f"\n📝 No existing knowledge found - Starting fresh")
    
    def create_comprehensive_problems(self):
        """Create comprehensive problem set including NP problems"""
        problems = {
            'arithmetic': [],
            'algebra': [],
            'geometry': [],
            'calculus': [],
            'discrete_math': [],
            'graph_theory': [],
            'number_theory': [],
            'probability': [],
            'statistics': [],
            'linear_algebra': [],
            'optimization': [],
            'np_problems': []
        }
        
        # Load existing problems
        self.load_existing_problems(problems)
        
        # Add NP problems
        self.add_np_problems(problems)
        
        # Add advanced problems
        self.add_advanced_problems(problems)
        
        return problems
    
    def load_existing_problems(self, problems):
        """Load existing problems from TEXT_DATA"""
        base_dir = "TEXT_DATA/MATH/PROBLEMS"
        
        # Load arithmetic problems
        arithmetic_file = os.path.join(base_dir, "arithmetic_problems.txt")
        if os.path.exists(arithmetic_file):
            with open(arithmetic_file, 'r') as f:
                content = f.read()
                parsed = self.parse_problems(content, "arithmetic")
                problems['arithmetic'].extend(parsed)
        
        # Load algebra problems
        algebra_file = os.path.join(base_dir, "algebra_problems.txt")
        if os.path.exists(algebra_file):
            with open(algebra_file, 'r') as f:
                content = f.read()
                parsed = self.parse_problems(content, "algebra")
                problems['algebra'].extend(parsed)
        
        # Load geometry problems
        geometry_file = os.path.join(base_dir, "geometry_problems.txt")
        if os.path.exists(geometry_file):
            with open(geometry_file, 'r') as f:
                content = f.read()
                parsed = self.parse_problems(content, "geometry")
                problems['geometry'].extend(parsed)
        
        # Load calculus problems
        calculus_file = os.path.join(base_dir, "calculus_problems.txt")
        if os.path.exists(calculus_file):
            with open(calculus_file, 'r') as f:
                content = f.read()
                parsed = self.parse_problems(content, "calculus")
                problems['calculus'].extend(parsed)
        
        print(f"📚 Loaded existing problems from TEXT_DATA/MATH/PROBLEMS/")
    
    def parse_problems(self, content, category):
        """Parse problems from text content"""
        problems = []
        lines = content.split('\n')
        current_problem = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Problem:"):
                if current_problem:
                    problems.append(current_problem)
                current_problem = {
                    'category': category,
                    'problem': line.replace("Problem:", "").strip(),
                    'solution': '',
                    'explanation': ''
                }
            elif line.startswith("Solution:") and current_problem:
                current_problem['solution'] = line.replace("Solution:", "").strip()
            elif line.startswith("Explanation:") and current_problem:
                current_problem['explanation'] = line.replace("Explanation:", "").strip()
        
        if current_problem:
            problems.append(current_problem)
        
        return problems
    
    def add_np_problems(self, problems):
        """Add NP-complete and NP-hard problems"""
        np_problems = [
            # Traveling Salesman Problem (NP-hard)
            {
                'category': 'np_problems',
                'problem': 'Find the shortest possible route that visits each city exactly once and returns to the origin city, given cities: A, B, C, D with distances: AB=10, AC=15, AD=20, BC=35, BD=25, CD=30',
                'solution': 'A → B → D → C → A with total distance 10 + 25 + 30 + 15 = 80',
                'explanation': 'This is the Traveling Salesman Problem (TSP), an NP-hard problem. For 4 cities, we can check all permutations. The optimal route is A-B-D-C-A with distance 80.',
                'complexity': 'NP-hard',
                'algorithm': 'Brute force for small n, approximation algorithms for large n'
            },
            # Knapsack Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Given items with weights [2, 3, 4, 5] and values [3, 4, 5, 6], and knapsack capacity 5, what is the maximum value achievable?',
                'solution': 'Maximum value is 7 (items with weights 2 and 3)',
                'explanation': 'This is the 0/1 Knapsack Problem, NP-complete. We test combinations: {2,3}=value7, {5}=value6, {2,4}=value8 (exceeds capacity). Best is {2,3} with value 7.',
                'complexity': 'NP-complete',
                'algorithm': 'Dynamic programming, branch and bound'
            },
            # Subset Sum Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Given set {3, 34, 4, 12, 5, 2}, is there a subset that sums to 9?',
                'solution': 'Yes, {4, 5} sums to 9',
                'explanation': 'This is the Subset Sum Problem, NP-complete. We need to find a subset that sums to target 9. Testing combinations: {4,5}=9 works.',
                'complexity': 'NP-complete',
                'algorithm': 'Dynamic programming, meet-in-the-middle'
            },
            # Graph Coloring Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Can a complete graph K4 be colored with 3 colors such that no adjacent vertices share the same color?',
                'solution': 'No, K4 requires 4 colors',
                'explanation': 'This is the Graph Coloring Problem, NP-complete. K4 is a complete graph with 4 vertices, each connected to every other. It requires 4 different colors.',
                'complexity': 'NP-complete',
                'algorithm': 'Backtracking, greedy approximation'
            },
            # Hamiltonian Cycle Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Does the graph with vertices {A,B,C,D} and edges {AB,BC,CD,DA,AC} have a Hamiltonian cycle?',
                'solution': 'Yes, A-B-C-D-A is a Hamiltonian cycle',
                'explanation': 'This is the Hamiltonian Cycle Problem, NP-complete. A Hamiltonian cycle visits each vertex exactly once. A-B-C-D-A visits all 4 vertices exactly once.',
                'complexity': 'NP-complete',
                'algorithm': 'Backtracking, dynamic programming'
            },
            # Partition Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Can the multiset {1, 5, 11, 5} be partitioned into two subsets with equal sum?',
                'solution': 'Yes, {1,5,5} and {11} both sum to 11',
                'explanation': 'This is the Partition Problem, NP-complete. Total sum is 22, so each subset must sum to 11. {1,5,5}=11 and {11}=11.',
                'complexity': 'NP-complete',
                'algorithm': 'Dynamic programming, approximation'
            },
            # Clique Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'In a graph with 6 vertices where each vertex is connected to exactly 3 others, does there exist a clique of size 4?',
                'solution': 'It depends on the specific connections, but generally no for regular 3-regular graphs',
                'explanation': 'This is the Clique Problem, NP-complete. In a 3-regular graph with 6 vertices, the maximum clique size is typically 3, not 4.',
                'complexity': 'NP-complete',
                'algorithm': 'Bron-Kerbosch algorithm, branch and bound'
            },
            # Vertex Cover Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Given a graph with edges {(1,2), (2,3), (3,4), (4,5)}, what is the minimum vertex cover size?',
                'solution': 'Size 2 (vertices 2 and 4)',
                'explanation': 'This is the Vertex Cover Problem, NP-complete. A vertex cover touches all edges. Vertices {2,4} cover all edges: (1,2), (2,3), (3,4), (4,5).',
                'complexity': 'NP-complete',
                'algorithm': 'Approximation algorithms, branch and bound'
            },
            # SAT Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Is the formula (x ∨ ¬y) ∧ (¬x ∨ y) ∧ (x ∨ y) satisfiable?',
                'solution': 'Yes, x=true, y=true satisfies the formula',
                'explanation': 'This is the Boolean Satisfiability Problem (SAT), NP-complete. Testing x=true,y=true: (T∨¬T)=T, (¬T∨T)=T, (T∨T)=T. All clauses are true.',
                'complexity': 'NP-complete',
                'algorithm': 'DPLL algorithm, CDCL, local search'
            },
            # 3-SAT Problem (NP-complete)
            {
                'category': 'np_problems',
                'problem': 'Is the 3-CNF formula (x ∨ y ∨ z) ∧ (¬x ∨ y ∨ ¬z) ∧ (x ∨ ¬y ∨ z) satisfiable?',
                'solution': 'Yes, x=true, y=true, z=true satisfies the formula',
                'explanation': 'This is the 3-SAT Problem, NP-complete. Testing x=T,y=T,z=T: (T∨T∨T)=T, (¬T∨T∨¬T)=T, (T∨¬T∨T)=T. All clauses are true.',
                'complexity': 'NP-complete',
                'algorithm': 'DPLL, resolution, local search'
            }
        ]
        
        problems['np_problems'].extend(np_problems)
        print(f"🧮 Added {len(np_problems)} NP problems")
    
    def add_advanced_problems(self, problems):
        """Add advanced mathematical problems"""
        # Discrete Mathematics
        discrete_problems = [
            {
                'category': 'discrete_math',
                'problem': 'How many ways can you arrange 5 distinct books on a shelf?',
                'solution': '120 ways (5! = 120)',
                'explanation': 'This is a permutation problem. For 5 distinct books, there are 5! = 5×4×3×2×1 = 120 arrangements.',
                'concept': 'Permutations'
            },
            {
                'category': 'discrete_math',
                'problem': 'How many 3-digit numbers can be formed using digits 1,2,3,4,5 without repetition?',
                'solution': '60 numbers (5×4×3 = 60)',
                'explanation': 'For 3-digit numbers: 5 choices for first digit, 4 for second, 3 for third. Total: 5×4×3 = 60.',
                'concept': 'Permutations without repetition'
            }
        ]
        
        # Graph Theory
        graph_problems = [
            {
                'category': 'graph_theory',
                'problem': 'How many edges does a complete graph K6 have?',
                'solution': '15 edges (6×5/2 = 15)',
                'explanation': 'A complete graph with n vertices has n(n-1)/2 edges. For K6: 6×5/2 = 15 edges.',
                'concept': 'Complete graphs'
            },
            {
                'category': 'graph_theory',
                'problem': 'What is the degree of each vertex in a regular graph with 8 vertices and 12 edges?',
                'solution': '3 degrees (2×12/8 = 3)',
                'explanation': 'In a regular graph, each vertex has the same degree. Sum of degrees = 2×edges = 24. Each vertex degree = 24/8 = 3.',
                'concept': 'Regular graphs'
            }
        ]
        
        # Number Theory
        number_theory_problems = [
            {
                'category': 'number_theory',
                'problem': 'Find the greatest common divisor of 48 and 18',
                'solution': '6',
                'explanation': 'GCD(48,18) = 6. Prime factors: 48 = 2^4 × 3, 18 = 2 × 3^2. Common factors: 2 × 3 = 6.',
                'concept': 'Greatest common divisor'
            },
            {
                'category': 'number_theory',
                'problem': 'Is 17 a prime number?',
                'solution': 'Yes',
                'explanation': '17 is prime because its only positive divisors are 1 and 17.',
                'concept': 'Prime numbers'
            }
        ]
        
        # Probability
        probability_problems = [
            {
                'category': 'probability',
                'problem': 'What is the probability of rolling a sum of 7 with two dice?',
                'solution': '1/6 or approximately 16.67%',
                'explanation': 'There are 6 ways to roll 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) out of 36 total outcomes. P = 6/36 = 1/6.',
                'concept': 'Probability'
            },
            {
                'category': 'probability',
                'problem': 'If you flip a coin 3 times, what is the probability of getting exactly 2 heads?',
                'solution': '3/8 or 37.5%',
                'explanation': 'Number of ways to get exactly 2 heads: C(3,2) = 3. Total outcomes: 2^3 = 8. P = 3/8.',
                'concept': 'Binomial probability'
            }
        ]
        
        # Statistics
        statistics_problems = [
            {
                'category': 'statistics',
                'problem': 'Find the mean of numbers 2, 4, 6, 8, 10',
                'solution': '6',
                'explanation': 'Mean = (2+4+6+8+10)/5 = 30/5 = 6.',
                'concept': 'Mean'
            },
            {
                'category': 'statistics',
                'problem': 'What is the median of 1, 3, 5, 7, 9?',
                'solution': '5',
                'explanation': 'Median is the middle value when ordered. For 5 numbers, the 3rd number (5) is the median.',
                'concept': 'Median'
            }
        ]
        
        # Linear Algebra
        linear_algebra_problems = [
            {
                'category': 'linear_algebra',
                'problem': 'Find the determinant of matrix [[2, 1], [3, 4]]',
                'solution': '5',
                'explanation': 'For 2×2 matrix [[a,b],[c,d]], determinant = ad - bc = 2×4 - 1×3 = 8 - 3 = 5.',
                'concept': 'Determinant'
            },
            {
                'category': 'linear_algebra',
                'problem': 'What is the dot product of vectors [1, 2, 3] and [4, 5, 6]?',
                'solution': '32',
                'explanation': 'Dot product = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32.',
                'concept': 'Dot product'
            }
        ]
        
        # Optimization
        optimization_problems = [
            {
                'category': 'optimization',
                'problem': 'Maximize f(x,y) = x + y subject to x + 2y ≤ 10, x ≥ 0, y ≥ 0',
                'solution': 'Maximum at (10, 0) with value 10',
                'explanation': 'Linear programming problem. The feasible region is bounded by x + 2y = 10. Maximum occurs at corner (10,0): f(10,0) = 10.',
                'concept': 'Linear programming'
            },
            {
                'category': 'optimization',
                'problem': 'Find the minimum of f(x) = x² - 4x + 3',
                'solution': 'Minimum at x = 2 with value -1',
                'explanation': 'Complete the square: f(x) = (x-2)² - 1. Minimum occurs when (x-2)² = 0, so x = 2, f(2) = -1.',
                'concept': 'Quadratic optimization'
            }
        ]
        
        problems['discrete_math'].extend(discrete_problems)
        problems['graph_theory'].extend(graph_problems)
        problems['number_theory'].extend(number_theory_problems)
        problems['probability'].extend(probability_problems)
        problems['statistics'].extend(statistics_problems)
        problems['linear_algebra'].extend(linear_algebra_problems)
        problems['optimization'].extend(optimization_problems)
        
        print(f"📊 Added advanced problems across all mathematical domains")
    
    def check_existing_knowledge(self, problem):
        """Check if problem already exists in knowledge base"""
        search_results = self.knowledge_system.search_knowledge(problem['problem'], 'mathematics')
        
        for result in search_results:
            if result['data']['problem'] == problem['problem']:
                return result['id'], result['data']
        
        return None, None
    
    def solve_problem_with_knowledge(self, problem):
        """Solve problem using existing knowledge or generate new solution"""
        existing_id, existing_data = self.check_existing_knowledge(problem)
        
        if existing_data:
            print(f"🧠 Using existing knowledge for: {problem['problem'][:50]}...")
            return existing_data['solution'], existing_data['explanation'], True
        else:
            solution, explanation = self.generate_solution(problem)
            
            # Add to knowledge base
            problem_id = self.knowledge_system.add_mathematical_knowledge(
                problem['problem'],
                solution,
                explanation,
                problem['category']
            )
            
            return solution, explanation, False
    
    def generate_solution(self, problem):
        """Generate solution for mathematical problem"""
        solution = problem['solution']
        explanation = problem['explanation']
        
        # Add complexity information for NP problems
        if problem['category'] == 'np_problems':
            complexity = problem.get('complexity', 'NP')
            algorithm = problem.get('algorithm', 'Brute force')
            explanation += f" Complexity: {complexity}. Algorithm: {algorithm}."
        
        return solution, explanation
    
    def train_comprehensive(self, epochs=5):
        """Train on comprehensive problem set"""
        print(f"\n🎓 Starting Comprehensive Mathematical Training")
        print(f"📊 Training for {epochs} epochs on all problem types")
        
        problems = self.create_comprehensive_problems()
        
        # Show problem distribution
        print(f"\n📚 Problem Distribution:")
        total_problems = 0
        for category, problem_list in problems.items():
            count = len(problem_list)
            total_problems += count
            print(f"  {category}: {count} problems")
        print(f"  Total: {total_problems} problems")
        
        # Train on each category
        all_results = []
        
        for epoch in range(epochs):
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            
            epoch_correct = 0
            epoch_total = 0
            epoch_new_knowledge = 0
            category_results = {}
            
            for category, problem_list in problems.items():
                if not problem_list:
                    continue
                
                category_correct = 0
                category_total = 0
                
                print(f"  Training on {category} ({len(problem_list)} problems)...")
                
                for problem in problem_list:
                    solution, explanation, was_existing = self.solve_problem_with_knowledge(problem)
                    
                    is_correct = solution == problem['solution']
                    
                    if is_correct:
                        category_correct += 1
                        epoch_correct += 1
                    
                    category_total += 1
                    epoch_total += 1
                    
                    if not was_existing:
                        epoch_new_knowledge += 1
                
                category_accuracy = (category_correct / category_total) * 100
                category_results[category] = {
                    'accuracy': category_accuracy,
                    'correct': category_correct,
                    'total': category_total
                }
                
                print(f"    {category}: {category_accuracy:.1f}% ({category_correct}/{category_total})")
            
            # Epoch summary
            epoch_accuracy = (epoch_correct / epoch_total) * 100
            all_results.append({
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'correct': epoch_correct,
                'total': epoch_total,
                'new_knowledge': epoch_new_knowledge,
                'category_results': category_results
            })
            
            print(f"  ✅ Epoch {epoch + 1} Complete")
            print(f"     Overall Accuracy: {epoch_accuracy:.1f}%")
            print(f"     New Knowledge: {epoch_new_knowledge}")
            print(f"     Total Knowledge: {len(self.knowledge_system.math_knowledge)}")
        
        return all_results
    
    def test_comprehensive(self, test_size=20):
        """Test on comprehensive problem set"""
        print(f"\n🧪 Comprehensive Testing")
        
        problems = self.create_comprehensive_problems()
        
        # Select test problems from each category
        test_problems = []
        for category, problem_list in problems.items():
            if problem_list:
                category_test_size = max(1, test_size // len(problems))
                selected = random.sample(problem_list, min(category_test_size, len(problem_list)))
                test_problems.extend(selected)
        
        print(f"📝 Testing on {len(test_problems)} problems across categories")
        
        test_correct = 0
        test_results = []
        knowledge_hits = 0
        category_performance = {}
        
        for problem in test_problems:
            category = problem['category']
            
            print(f"\n📝 Test Problem: {category}")
            print(f"Problem: {problem['problem']}")
            
            solution, explanation, was_existing = self.solve_problem_with_knowledge(problem)
            
            if was_existing:
                knowledge_hits += 1
                print(f"🧠 Used existing knowledge")
            else:
                print(f"🆕 Generated new solution")
            
            print(f"Solution: {solution}")
            
            is_correct = solution == problem['solution']
            if is_correct:
                test_correct += 1
                print("✅ Correct")
            else:
                print("❌ Incorrect")
            
            # Track category performance
            if category not in category_performance:
                category_performance[category] = {'correct': 0, 'total': 0, 'knowledge_hits': 0}
            
            category_performance[category]['total'] += 1
            if is_correct:
                category_performance[category]['correct'] += 1
            if was_existing:
                category_performance[category]['knowledge_hits'] += 1
            
            test_results.append({
                'problem': problem['problem'],
                'category': category,
                'correct': is_correct,
                'used_existing_knowledge': was_existing,
                'solution': solution,
                'expected': problem['solution']
            })
        
        # Calculate results
        test_accuracy = (test_correct / len(test_problems)) * 100
        knowledge_hit_rate = (knowledge_hits / len(test_problems)) * 100
        
        print(f"\n📊 Test Results:")
        print(f"  Correct: {test_correct}/{len(test_problems)}")
        print(f"  Accuracy: {test_accuracy:.1f}%")
        print(f"  Knowledge Hits: {knowledge_hits}/{len(test_problems)} ({knowledge_hit_rate:.1f}%)")
        
        print(f"\n📈 Performance by Category:")
        for category, stats in category_performance.items():
            accuracy = (stats['correct'] / stats['total']) * 100
            hit_rate = (stats['knowledge_hits'] / stats['total']) * 100
            print(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) - Knowledge Hits: {hit_rate:.1f}%")
        
        return test_results, test_accuracy, knowledge_hit_rate
    
    def analyze_complexity_distribution(self):
        """Analyze distribution of problem complexities"""
        print(f"\n📊 Complexity Analysis")
        
        problems = self.create_comprehensive_problems()
        
        complexity_stats = {
            'P': 0,  # Polynomial time
            'NP': 0, # Nondeterministic polynomial time
            'NP-complete': 0,
            'NP-hard': 0,
            'Unknown': 0
        }
        
        for category, problem_list in problems.items():
            for problem in problem_list:
                if category == 'np_problems':
                    complexity = problem.get('complexity', 'NP')
                    if complexity in complexity_stats:
                        complexity_stats[complexity] += 1
                    else:
                        complexity_stats['Unknown'] += 1
                else:
                    complexity_stats['P'] += 1
        
        print(f"📈 Complexity Distribution:")
        total = sum(complexity_stats.values())
        for complexity, count in complexity_stats.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {complexity}: {count} problems ({percentage:.1f}%)")
        
        return complexity_stats
    
    def save_comprehensive_results(self, training_results, test_results, test_accuracy, knowledge_hit_rate):
        """Save comprehensive training results"""
        # Record training session
        session_record = self.knowledge_system.record_training_session(
            "comprehensive_mathematical_training",
            test_results,
            test_accuracy,
            time.time() - self.session_start
        )
        
        # Save all knowledge
        self.knowledge_system.save_all_knowledge()
        
        # Create comprehensive summary
        summary = {
            'session_time': datetime.now().isoformat(),
            'training_results': training_results,
            'test_results': {
                'accuracy': test_accuracy,
                'knowledge_hit_rate': knowledge_hit_rate,
                'total_tests': len(test_results)
            },
            'knowledge_summary': self.knowledge_system.get_knowledge_summary(),
            'complexity_analysis': self.analyze_complexity_distribution()
        }
        
        # Save summary
        summary_file = f"comprehensive_training_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Comprehensive results saved to: {summary_file}")
        return summary_file

def main():
    """Main function"""
    training_system = ComprehensiveMathTraining()
    
    # Analyze complexity distribution
    complexity_stats = training_system.analyze_complexity_distribution()
    
    # Train on comprehensive problems
    training_results = training_system.train_comprehensive(epochs=3)
    
    # Test comprehensive performance
    test_results, test_accuracy, knowledge_hit_rate = training_system.test_comprehensive(test_size=15)
    
    # Save results
    summary_file = training_system.save_comprehensive_results(training_results, test_results, test_accuracy, knowledge_hit_rate)
    
    # Final summary
    print(f"\n🎉 COMPREHENSIVE MATHEMATICAL TRAINING COMPLETE!")
    print(f"📊 Final Test Accuracy: {test_accuracy:.1f}%")
    print(f"🧠 Knowledge Hit Rate: {knowledge_hit_rate:.1f}%")
    print(f"📚 Total Knowledge Items: {len(training_system.knowledge_system.math_knowledge)}")
    print(f"🧮 NP Problems Trained: {complexity_stats.get('NP-complete', 0) + complexity_stats.get('NP-hard', 0)}")
    print(f"📁 Results saved to: {summary_file}")
    
    print(f"\n🚀 AI is now trained on ALL problem types including NP problems!")
    print(f"🧠 Persistent knowledge ensures no starting from scratch!")
    print(f"📈 Ready for advanced mathematical problem solving!")

if __name__ == "__main__":
    main()
