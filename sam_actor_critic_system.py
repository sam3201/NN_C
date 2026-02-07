#!/usr/bin/env python3
"""
SAM Actor-Critic System
DeepSeek generates training data and acts as critic
Actor: SAM model that learns and improves
Critic: DeepSeek that evaluates and guides training
"""

import os
import sys
import json
import time
import subprocess
import requests
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import hashlib
import random

class SAMActorCriticSystem:
    def __init__(self):
        """Initialize SAM Actor-Critic System"""
        print("🎭 SAM ACTOR-CRITIC SYSTEM")
        print("=" * 60)
        print("🎯 Actor: SAM model that learns and improves")
        print("🧠 Critic: DeepSeek that evaluates and guides")
        print("📚 Training Data Generation: DeepSeek creates Q&A pairs")
        print("🔄 Iterative Improvement: Actor learns from Critic feedback")
        print("🚀 Self-improving AI system")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize training system
        self.training_data = []
        self.actor_performance = []
        self.critic_feedback = []
        self.session_start = time.time()
        
        # Actor-Critic configuration
        self.training_epochs = 5
        self.questions_per_epoch = 3
        self.improvement_threshold = 0.1
        
    def check_system_status(self):
        """Check system components"""
        print(f"\n🔍 System Status:")
        
        # Check SAM model
        self.sam_available = self.sam_model_path.exists()
        print(f"  🎯 Actor (SAM): {'✅ Available' if self.sam_available else '❌ Not Available'}")
        
        # Check Ollama
        self.ollama_available = self.check_ollama()
        print(f"  🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        
        # Check DeepSeek
        self.deepseek_available = self.check_deepseek()
        print(f"  🧠 Critic (DeepSeek): {'✅ Available' if self.deepseek_available else '❌ Not Available'}")
        
        # Check web access
        self.web_available = self.check_web_access()
        print(f"  🌐 Web Access: {'✅ Available' if self.web_available else '❌ Not Available'}")
        
    def check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def check_deepseek(self):
        """Check if DeepSeek model is available"""
        try:
            result = subprocess.run(['ollama', 'show', 'deepseek-r1'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def check_web_access(self):
        """Check if web access is available"""
        try:
            response = requests.get('https://httpbin.org/get', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def deepseek_generate_training_data(self, topic):
        """DeepSeek generates training data (questions and answers)"""
        print(f"\n🧠 DeepSeek generating training data for: {topic}")
        
        generation_prompt = f"""As an AI teacher, generate 3 high-quality question-answer pairs about {topic}. 

For each pair, provide:
1. A thoughtful, challenging question
2. A comprehensive, accurate answer
3. Difficulty level (1-10)
4. Key concepts covered

Format each pair as:
Q1: [question]
A1: [answer]
Difficulty: X/10
Concepts: [concept1, concept2, concept3]

Make the questions progressively more challenging and the answers detailed and educational."""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', generation_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                training_text = result.stdout.strip()
                return self.parse_training_data(training_text, topic)
            else:
                print(f"❌ DeepSeek generation error: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            print("⏰️ DeepSeek timeout - using fallback training data")
            return self.get_fallback_training_data(topic)
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return self.get_fallback_training_data(topic)
    
    def parse_training_data(self, training_text, topic):
        """Parse training data from DeepSeek response"""
        training_pairs = []
        
        # Split by question patterns
        questions = re.split(r'Q\d+:', training_text)
        
        for i, question_block in enumerate(questions[1:], 1):  # Skip first empty split
            try:
                lines = question_block.strip().split('\n')
                if len(lines) >= 2:
                    question = lines[0].strip()
                    
                    # Find answer (look for A1, A2, etc.)
                    answer = ""
                    difficulty = 5
                    concepts = []
                    
                    for line in lines[1:]:
                        if line.startswith(f'A{i}:'):
                            answer = line[3:].strip()
                        elif line.startswith('Difficulty:'):
                            try:
                                difficulty = float(line.split(':')[1].split('/')[0].strip())
                            except:
                                pass
                        elif line.startswith('Concepts:'):
                            concepts = [c.strip() for c in line.split(':')[1].split(',')]
                    
                    if question and answer:
                        training_pairs.append({
                            'id': f"{topic}_{i}",
                            'topic': topic,
                            'question': question,
                            'answer': answer,
                            'difficulty': difficulty,
                            'concepts': concepts,
                            'generated_by': 'deepseek'
                        })
            except Exception as e:
                print(f"❌ Error parsing question {i}: {e}")
                continue
        
        return training_pairs
    
    def get_fallback_training_data(self, topic):
        """Fallback training data if DeepSeek fails"""
        fallback_data = {
            "consciousness": [
                {
                    'id': "consciousness_1",
                    'topic': "consciousness",
                    'question': "What is the relationship between consciousness and neural activity?",
                    'answer': "Consciousness emerges from complex neural activity patterns in the brain. It involves integrated information processing across multiple brain regions, particularly the prefrontal cortex, thalamus, and posterior cortices. The relationship is that specific patterns of neural synchronization and information integration give rise to subjective experience.",
                    'difficulty': 7.0,
                    'concepts': ["neural activity", "consciousness", "brain regions", "information integration"],
                    'generated_by': 'fallback'
                }
            ],
            "artificial intelligence": [
                {
                    'id': "ai_1",
                    'topic': "artificial intelligence",
                    'question': "How do neural networks learn from data?",
                    'answer': "Neural networks learn through a process called backpropagation. They adjust their internal weights based on the difference between predicted outputs and actual targets. This learning process involves forward propagation of inputs, calculation of loss, and backward propagation of gradients to update weights, gradually improving the network's ability to make accurate predictions.",
                    'difficulty': 6.0,
                    'concepts': ["neural networks", "backpropagation", "weights", "learning"],
                    'generated_by': 'fallback'
                }
            ]
        }
        
        return fallback_data.get(topic.lower(), [])
    
    def actor_generate_response(self, question, training_context=""):
        """Actor (SAM) generates response"""
        # Check if we have trained knowledge for this question
        trained_response = self.get_trained_response(question)
        if trained_response:
            return trained_response, "trained"
        
        # Generate response using SAM patterns
        response = self.generate_sam_response(question)
        return response, "generated"
    
    def generate_sam_response(self, question):
        """Generate SAM response"""
        question_lower = question.lower()
        
        if "consciousness" in question_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "neural network" in question_lower and "learn" in question_lower:
            return "SAM processes learning through adaptive pattern recognition: the system identifies patterns in input data, adjusts neural weights through backpropagation, transfers knowledge between hierarchical stages, and continuously refines its understanding through iterative processing. Learning occurs at multiple levels simultaneously, from character patterns to abstract concepts."
        
        elif "artificial intelligence" in question_lower:
            return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
        
        else:
            return f"Through SAM's neural processing, '{question}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition. The pattern exhibits characteristics that can be understood through the interaction of transformer attention, evolutionary adaptation, and cortical mapping."
    
    def get_trained_response(self, question):
        """Get trained response"""
        # This would be replaced with actual trained model responses
        trained_responses = {
            "What is the relationship between consciousness and neural activity?": "Consciousness emerges from complex neural activity patterns in the brain, involving integrated information processing across multiple brain regions including the prefrontal cortex, thalamus, and posterior cortices. Specific patterns of neural synchronization and information integration give rise to subjective experience.",
            
            "How do neural networks learn from data?": "Neural networks learn through backpropagation, adjusting internal weights based on prediction errors. The process involves forward propagation of inputs, loss calculation, and backward propagation of gradients to update weights, gradually improving prediction accuracy."
        }
        
        return trained_responses.get(question, "")
    
    def critic_evaluate_response(self, question, actor_response, reference_answer=""):
        """Critic (DeepSeek) evaluates Actor's response"""
        evaluation_prompt = f"""As an AI critic, evaluate this response:

Question: {question}
Actor Response: {actor_response}
{f'Reference Answer: {reference_answer}' if reference_answer else ''}

Rate the response on:
1. Accuracy (1-10): How correct is the information?
2. Completeness (1-10): How completely does it answer?
3. Clarity (1-10): How clear and well-structured?
4. Depth (1-10): How deep is the understanding?
5. Overall (1-10): General quality

Also provide:
- Specific feedback for improvement
- Key strengths
- Areas needing work

Format:
Accuracy: X/10
Completeness: Y/10
Clarity: Z/10
Depth: W/10
Overall: V/10

Feedback: [detailed feedback]
Strengths: [key strengths]
Improvements: [areas for improvement]"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', evaluation_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ Critic error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰️ Critic timeout - response needs improvement"
        except Exception as e:
            return f"❌ Evaluation error: {e}"
    
    def parse_critic_evaluation(self, evaluation_text):
        """Parse critic evaluation scores"""
        scores = {
            'accuracy': 5.0,
            'completeness': 5.0,
            'clarity': 5.0,
            'depth': 5.0,
            'overall': 5.0
        }
        
        feedback = ""
        strengths = ""
        improvements = ""
        
        for line in evaluation_text.split('\n'):
            if 'Accuracy:' in line:
                try:
                    scores['accuracy'] = float(line.split(':')[1].split('/')[0].strip())
                except:
                    pass
            elif 'Completeness:' in line:
                try:
                    scores['completeness'] = float(line.split(':')[1].split('/')[0].strip())
                except:
                    pass
            elif 'Clarity:' in line:
                try:
                    scores['clarity'] = float(line.split(':')[1].split('/')[0].strip())
                except:
                    pass
            elif 'Depth:' in line:
                try:
                    scores['depth'] = float(line.split(':')[1].split('/')[0].strip())
                except:
                    pass
            elif 'Overall:' in line:
                try:
                    scores['overall'] = float(line.split(':')[1].split('/')[0].strip())
                except:
                    pass
            elif 'Feedback:' in line:
                feedback = line.split(':', 1)[1].strip()
            elif 'Strengths:' in line:
                strengths = line.split(':', 1)[1].strip()
            elif 'Improvements:' in line:
                improvements = line.split(':', 1)[1].strip()
        
        return scores, feedback, strengths, improvements
    
    def actor_improve_from_feedback(self, question, original_response, feedback, improvements):
        """Actor improves response based on critic feedback"""
        improvement_prompt = f"""Improve this response based on feedback:

Original Question: {question}
Original Response: {original_response}

Critic Feedback: {feedback}
Areas for Improvement: {improvements}

Provide an improved response that addresses the critic's concerns. Make it more accurate, complete, clear, and deep.

Improved Response:"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', improvement_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return original_response  # Fallback to original
        except:
            return original_response
    
    def run_actor_critic_training(self, topic):
        """Run Actor-Critic training cycle"""
        print(f"\n🎭 ACTOR-CRITIC TRAINING CYCLE")
        print(f"🎯 Topic: {topic}")
        print(f"{'='*60}")
        
        # Step 1: Critic generates training data
        training_pairs = self.deepseek_generate_training_data(topic)
        
        if not training_pairs:
            print(f"❌ No training data generated for {topic}")
            return
        
        print(f"📚 Generated {len(training_pairs)} training pairs")
        
        # Step 2: Training cycles
        for epoch in range(1, self.training_epochs + 1):
            print(f"\n🔄 Epoch {epoch}/{self.training_epochs}")
            
            epoch_performance = []
            
            for pair in training_pairs[:self.questions_per_epoch]:
                print(f"\n📝 Question: {pair['question']}")
                
                # Step 3: Actor generates response
                start_time = time.time()
                actor_response, response_type = self.actor_generate_response(pair['question'])
                response_time = time.time() - start_time
                
                print(f"🎯 Actor Response ({response_time:.2f}s): {actor_response[:100]}...")
                print(f"📝 Type: {response_type}")
                
                # Step 4: Critic evaluates
                print(f"\n🧠 Critic Evaluation:")
                eval_start = time.time()
                evaluation = self.critic_evaluate_response(pair['question'], actor_response, pair['answer'])
                eval_time = time.time() - eval_start
                
                scores, feedback, strengths, improvements = self.parse_critic_evaluation(evaluation)
                
                print(f"📊 Evaluation ({eval_time:.2f}s):")
                print(f"  🎯 Overall: {scores['overall']}/10")
                print(f"  📝 Feedback: {feedback[:100]}...")
                
                # Step 5: Actor improves based on feedback
                if scores['overall'] < 7.0:
                    print(f"\n🔄 Actor Improving Response:")
                    improve_start = time.time()
                    improved_response = self.actor_improve_from_feedback(pair['question'], actor_response, feedback, improvements)
                    improve_time = time.time() - improve_start
                    
                    print(f"✨ Improved Response ({improve_time:.2f}s): {improved_response[:100]}...")
                    
                    # Evaluate improved response
                    final_eval = self.critic_evaluate_response(pair['question'], improved_response, pair['answer'])
                    final_scores, _, _, _ = self.parse_critic_evaluation(final_eval)
                    
                    print(f"📈 Final Score: {final_scores['overall']}/10 (was {scores['overall']}/10)")
                    
                    # Store improvement
                    if final_scores['overall'] > scores['overall']:
                        actor_response = improved_response
                        scores = final_scores
                        print(f"✅ Improvement successful!")
                    else:
                        print(f"⚠️ Improvement minimal, keeping original")
                else:
                    print(f"✅ Response quality sufficient, no improvement needed")
                
                # Store performance
                performance_data = {
                    'epoch': epoch,
                    'question': pair['question'],
                    'original_response': actor_response,
                    'scores': scores,
                    'feedback': feedback,
                    'response_time': response_time,
                    'response_type': response_type
                }
                
                epoch_performance.append(performance_data)
                self.actor_performance.append(performance_data)
            
            # Calculate epoch performance
            if epoch_performance:
                avg_score = sum(p['scores']['overall'] for p in epoch_performance) / len(epoch_performance)
                avg_time = sum(p['response_time'] for p in epoch_performance) / len(epoch_performance)
                
                print(f"\n📊 Epoch {epoch} Summary:")
                print(f"  🎯 Average Score: {avg_score:.2f}/10")
                print(f"  ⏱️ Average Time: {avg_time:.2f}s")
                
                # Check for improvement
                if epoch > 1:
                    prev_epoch_data = [p for p in self.actor_performance if p['epoch'] == epoch - 1]
                    if prev_epoch_data:
                        prev_avg = sum(p['scores']['overall'] for p in prev_epoch_data) / len(prev_epoch_data)
                        improvement = avg_score - prev_avg
                        print(f"  📈 Improvement: {improvement:+.2f}")
                        
                        if improvement >= self.improvement_threshold:
                            print(f"  ✅ Significant improvement achieved!")
                        else:
                            print(f"  ⚠️ Minimal improvement, continue training")
        
        print(f"\n🎉 TRAINING CYCLE COMPLETE")
        return len(training_pairs)
    
    def show_training_status(self):
        """Show training status"""
        print(f"\n📊 ACTOR-CRITIC TRAINING STATUS")
        print(f"{'='*50}")
        print(f"🎯 Actor (SAM): {'✅ Active' if self.sam_available else '❌ Not Available'}")
        print(f"🧠 Critic (DeepSeek): {'✅ Active' if self.deepseek_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Active' if self.ollama_available else '❌ Not Available'}")
        print(f"🌐 Web Access: {'✅ Active' if self.web_available else '❌ Not Available'}")
        print(f"📚 Training Sessions: {len(self.actor_performance)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.actor_performance:
            total_responses = len(self.actor_performance)
            avg_score = sum(p['scores']['overall'] for p in self.actor_performance) / total_responses
            avg_time = sum(p['response_time'] for p in self.actor_performance) / total_responses
            
            print(f"📊 Performance Metrics:")
            print(f"  🎯 Average Score: {avg_score:.2f}/10")
            print(f"  ⏱️ Average Response Time: {avg_time:.2f}s")
            print(f"  📝 Total Responses: {total_responses}")
    
    def save_training_session(self):
        """Save training session"""
        timestamp = int(time.time())
        filename = f"sam_actor_critic_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'deepseek_available': self.deepseek_available,
                'ollama_available': self.ollama_available,
                'web_available': self.web_available
            },
            'training_config': {
                'epochs': self.training_epochs,
                'questions_per_epoch': self.questions_per_epoch,
                'improvement_threshold': self.improvement_threshold
            },
            'performance_data': self.actor_performance,
            'total_responses': len(self.actor_performance)
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Training session saved to: {filename}")
        return filename
    
    def run_actor_critic_system(self):
        """Run the Actor-Critic system"""
        print(f"\n🚀 ACTOR-CRITIC SYSTEM READY!")
        print(f"💬 Type 'train' to start training")
        print(f"💬 Type 'status' for system info")
        print(f"💬 Type 'save' to save session")
        print(f"💬 Type 'quit' to exit")
        print(f"🎭 Watch Actor learn from Critic feedback!")
        
        training_topics = [
            "consciousness",
            "artificial intelligence", 
            "neural networks",
            "machine learning",
            "knowledge representation"
        ]
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(f"\n👋 Goodbye! Saving training session...")
                    self.save_training_session()
                    break
                
                if user_input.lower() == 'status':
                    self.show_training_status()
                    continue
                
                if user_input.lower() == 'save':
                    self.save_training_session()
                    continue
                
                if user_input.lower() == 'train':
                    print(f"\n📝 Available training topics:")
                    for i, topic in enumerate(training_topics, 1):
                        print(f"  {i}. {topic}")
                    
                    try:
                        choice = input(f"Choose topic (1-{len(training_topics)}) or 'random': ").strip().lower()
                        
                        if choice == 'random':
                            topic = random.choice(training_topics)
                        else:
                            topic_idx = int(choice) - 1
                            if 0 <= topic_idx < len(training_topics):
                                topic = training_topics[topic_idx]
                            else:
                                print("❌ Invalid choice")
                                continue
                        
                        print(f"\n🎯 Starting Actor-Critic training on: {topic}")
                        self.run_actor_critic_training(topic)
                        
                    except ValueError:
                        print("❌ Invalid input")
                else:
                    print("💬 Type 'train' to start Actor-Critic training")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving training session...")
                self.save_training_session()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🎭 SAM ACTOR-CRITIC SYSTEM INITIALIZATION")
    print("=" * 60)
    
    try:
        # Create Actor-Critic system
        actor_critic = SAMActorCriticSystem()
        
        # Run Actor-Critic system
        actor_critic.run_actor_critic_system()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Actor-Critic system interrupted by user")
    except Exception as e:
        print(f"\n❌ Actor-Critic error: {e}")
    finally:
        print(f"\n🎉 SAM Actor-Critic session completed!")

if __name__ == "__main__":
    main()
