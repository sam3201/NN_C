#!/usr/bin/env python3
"""
Persistent Knowledge System for AI
Loads and remembers all previous training sessions
Builds upon existing knowledge without starting from scratch
"""

import os
import sys
import time
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

class PersistentKnowledgeSystem:
    def __init__(self, knowledge_base_dir="KNOWLEDGE_BASE"):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.knowledge_base_dir.mkdir(exist_ok=True)
        
        # Knowledge storage files
        self.math_knowledge_file = self.knowledge_base_dir / "math_knowledge.pkl"
        self.protein_knowledge_file = self.knowledge_base_dir / "protein_knowledge.pkl"
        self.concept_knowledge_file = self.knowledge_base_dir / "concept_knowledge.pkl"
        self.training_history_file = self.knowledge_base_dir / "training_history.json"
        self.knowledge_index_file = self.knowledge_base_dir / "knowledge_index.json"
        
        # Initialize knowledge storage
        self.math_knowledge = {}
        self.protein_knowledge = {}
        self.concept_knowledge = {}
        self.training_history = []
        self.knowledge_index = {}
        
        # Load existing knowledge
        self.load_all_knowledge()
        
        print("🧠 Persistent Knowledge System Initialized")
        print(f"📁 Knowledge Base: {self.knowledge_base_dir}")
        print(f"📊 Knowledge Files: {len(self.get_knowledge_files())} files loaded")
    
    def get_knowledge_files(self):
        """Get list of all knowledge files"""
        return list(self.knowledge_base_dir.glob("*.pkl")) + list(self.knowledge_base_dir.glob("*.json"))
    
    def load_all_knowledge(self):
        """Load all existing knowledge from persistent storage"""
        print("🔍 Loading Persistent Knowledge...")
        
        # Load mathematical knowledge
        if self.math_knowledge_file.exists():
            try:
                with open(self.math_knowledge_file, 'rb') as f:
                    self.math_knowledge = pickle.load(f)
                print(f"✅ Loaded {len(self.math_knowledge)} mathematical knowledge items")
            except Exception as e:
                print(f"⚠️  Error loading math knowledge: {e}")
                self.math_knowledge = {}
        else:
            print("📝 No existing math knowledge found - starting fresh")
        
        # Load protein folding knowledge
        if self.protein_knowledge_file.exists():
            try:
                with open(self.protein_knowledge_file, 'rb') as f:
                    self.protein_knowledge = pickle.load(f)
                print(f"✅ Loaded {len(self.protein_knowledge)} protein knowledge items")
            except Exception as e:
                print(f"⚠️  Error loading protein knowledge: {e}")
                self.protein_knowledge = {}
        else:
            print("📝 No existing protein knowledge found - starting fresh")
        
        # Load concept knowledge
        if self.concept_knowledge_file.exists():
            try:
                with open(self.concept_knowledge_file, 'rb') as f:
                    self.concept_knowledge = pickle.load(f)
                print(f"✅ Loaded {len(self.concept_knowledge)} concept knowledge items")
            except Exception as e:
                print(f"⚠️  Error loading concept knowledge: {e}")
                self.concept_knowledge = {}
        else:
            print("📝 No existing concept knowledge found - starting fresh")
        
        # Load training history
        if self.training_history_file.exists():
            try:
                with open(self.training_history_file, 'r') as f:
                    self.training_history = json.load(f)
                print(f"✅ Loaded {len(self.training_history)} training sessions")
            except Exception as e:
                print(f"⚠️  Error loading training history: {e}")
                self.training_history = []
        else:
            print("📝 No existing training history found - starting fresh")
        
        # Load knowledge index
        if self.knowledge_index_file.exists():
            try:
                with open(self.knowledge_index_file, 'r') as f:
                    self.knowledge_index = json.load(f)
                print(f"✅ Loaded knowledge index with {len(self.knowledge_index)} categories")
            except Exception as e:
                print(f"⚠️  Error loading knowledge index: {e}")
                self.knowledge_index = {}
        else:
            print("📝 No existing knowledge index found - starting fresh")
        
        return True
    
    def save_all_knowledge(self):
        """Save all knowledge to persistent storage"""
        print("💾 Saving Knowledge to Persistent Storage...")
        
        try:
            # Save mathematical knowledge
            with open(self.math_knowledge_file, 'wb') as f:
                pickle.dump(self.math_knowledge, f)
            print(f"✅ Saved {len(self.math_knowledge)} mathematical knowledge items")
            
            # Save protein folding knowledge
            with open(self.protein_knowledge_file, 'wb') as f:
                pickle.dump(self.protein_knowledge, f)
            print(f"✅ Saved {len(self.protein_knowledge)} protein knowledge items")
            
            # Save concept knowledge
            with open(self.concept_knowledge_file, 'wb') as f:
                pickle.dump(self.concept_knowledge, f)
            print(f"✅ Saved {len(self.concept_knowledge)} concept knowledge items")
            
            # Save training history
            with open(self.training_history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"✅ Saved {len(self.training_history)} training sessions")
            
            # Save knowledge index
            with open(self.knowledge_index_file, 'w') as f:
                json.dump(self.knowledge_index, f, indent=2)
            print(f"✅ Saved knowledge index with {len(self.knowledge_index)} categories")
            
            return True
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
            return False
    
    def add_mathematical_knowledge(self, problem, solution, explanation, category):
        """Add mathematical knowledge to persistent storage"""
        # Create unique ID for problem
        problem_id = hashlib.md5(f"{problem}{solution}".encode()).hexdigest()
        
        self.math_knowledge[problem_id] = {
            'problem': problem,
            'solution': solution,
            'explanation': explanation,
            'category': category,
            'timestamp': time.time(),
            'access_count': 0,
            'success_rate': 1.0
        }
        
        # Update knowledge index
        if 'mathematics' not in self.knowledge_index:
            self.knowledge_index['mathematics'] = {
                'categories': {},
                'total_items': 0,
                'last_updated': time.time()
            }
        
        if category not in self.knowledge_index['mathematics']['categories']:
            self.knowledge_index['mathematics']['categories'][category] = 0
        
        self.knowledge_index['mathematics']['categories'][category] += 1
        self.knowledge_index['mathematics']['total_items'] += 1
        self.knowledge_index['mathematics']['last_updated'] = time.time()
        
        print(f"📚 Added mathematical knowledge: {category} - {problem[:50]}...")
        return problem_id
    
    def add_protein_knowledge(self, sequence, structure, energy, properties):
        """Add protein folding knowledge to persistent storage"""
        # Create unique ID for sequence
        sequence_id = hashlib.md5(sequence.encode()).hexdigest()
        
        self.protein_knowledge[sequence_id] = {
            'sequence': sequence,
            'structure': structure,
            'energy': energy,
            'properties': properties,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        # Update knowledge index
        if 'protein_folding' not in self.knowledge_index:
            self.knowledge_index['protein_folding'] = {
                'categories': {},
                'total_items': 0,
                'last_updated': time.time()
            }
        
        structure_type = properties.get('dominant_structure', 'unknown')
        if structure_type not in self.knowledge_index['protein_folding']['categories']:
            self.knowledge_index['protein_folding']['categories'][structure_type] = 0
        
        self.knowledge_index['protein_folding']['categories'][structure_type] += 1
        self.knowledge_index['protein_folding']['total_items'] += 1
        self.knowledge_index['protein_folding']['last_updated'] = time.time()
        
        print(f"🧬 Added protein knowledge: {structure_type} - {sequence[:20]}...")
        return sequence_id
    
    def add_concept_knowledge(self, concept, definition, examples, domain):
        """Add conceptual knowledge to persistent storage"""
        # Create unique ID for concept
        concept_id = hashlib.md5(f"{concept}{domain}".encode()).hexdigest()
        
        self.concept_knowledge[concept_id] = {
            'concept': concept,
            'definition': definition,
            'examples': examples,
            'domain': domain,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        # Update knowledge index
        if 'concepts' not in self.knowledge_index:
            self.knowledge_index['concepts'] = {
                'categories': {},
                'total_items': 0,
                'last_updated': time.time()
            }
        
        if domain not in self.knowledge_index['concepts']['categories']:
            self.knowledge_index['concepts']['categories'][domain] = 0
        
        self.knowledge_index['concepts']['categories'][domain] += 1
        self.knowledge_index['concepts']['total_items'] += 1
        self.knowledge_index['concepts']['last_updated'] = time.time()
        
        print(f"📖 Added concept knowledge: {domain} - {concept}")
        return concept_id
    
    def record_training_session(self, session_type, results, accuracy, duration):
        """Record training session in persistent history"""
        session_record = {
            'session_type': session_type,
            'timestamp': time.time(),
            'date': datetime.now().isoformat(),
            'results': results,
            'accuracy': accuracy,
            'duration': duration,
            'knowledge_added': len(results) if isinstance(results, list) else 1
        }
        
        self.training_history.append(session_record)
        
        # Keep only last 100 sessions
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
        
        print(f"📝 Recorded training session: {session_type} - Accuracy: {accuracy:.1f}%")
        return session_record
    
    def get_knowledge_summary(self):
        """Get summary of all knowledge"""
        summary = {
            'mathematical_knowledge': len(self.math_knowledge),
            'protein_knowledge': len(self.protein_knowledge),
            'concept_knowledge': len(self.concept_knowledge),
            'training_sessions': len(self.training_history),
            'knowledge_categories': len(self.knowledge_index),
            'total_knowledge_items': len(self.math_knowledge) + len(self.protein_knowledge) + len(self.concept_knowledge)
        }
        
        return summary
    
    def search_knowledge(self, query, domain=None):
        """Search knowledge base for relevant information"""
        results = []
        query_lower = query.lower()
        
        # Search mathematical knowledge
        for item_id, item in self.math_knowledge.items():
            if domain is None or domain == 'mathematics':
                if (query_lower in item['problem'].lower() or 
                    query_lower in item['solution'].lower() or
                    query_lower in item['explanation'].lower()):
                    results.append({
                        'type': 'mathematical',
                        'id': item_id,
                        'data': item
                    })
        
        # Search protein knowledge
        for item_id, item in self.protein_knowledge.items():
            if domain is None or domain == 'protein_folding':
                if (query_lower in item['sequence'].lower() or
                    query_lower in str(item['properties']).lower()):
                    results.append({
                        'type': 'protein_folding',
                        'id': item_id,
                        'data': item
                    })
        
        # Search concept knowledge
        for item_id, item in self.concept_knowledge.items():
            if domain is None or domain == 'concepts':
                if (query_lower in item['concept'].lower() or
                    query_lower in item['definition'].lower() or
                    any(query_lower in str(ex).lower() for ex in item.get('examples', []))):
                    results.append({
                        'type': 'concept',
                        'id': item_id,
                        'data': item
                    })
        
        return results
    
    def get_learning_progress(self):
        """Get learning progress over time"""
        if not self.training_history:
            return {'message': 'No training history available'}
        
        progress = {
            'total_sessions': len(self.training_history),
            'average_accuracy': sum(s['accuracy'] for s in self.training_history) / len(self.training_history),
            'recent_sessions': self.training_history[-10:],
            'improvement_trend': self.calculate_improvement_trend(),
            'knowledge_growth': self.calculate_knowledge_growth()
        }
        
        return progress
    
    def calculate_improvement_trend(self):
        """Calculate improvement trend from training history"""
        if len(self.training_history) < 2:
            return 'insufficient_data'
        
        recent_sessions = self.training_history[-10:]
        older_sessions = self.training_history[-20:-10] if len(self.training_history) >= 20 else self.training_history[:-10]
        
        if not older_sessions:
            return 'insufficient_data'
        
        recent_avg = sum(s['accuracy'] for s in recent_sessions) / len(recent_sessions)
        older_avg = sum(s['accuracy'] for s in older_sessions) / len(older_sessions)
        
        if recent_avg > older_avg + 5:
            return 'improving'
        elif recent_avg < older_avg - 5:
            return 'declining'
        else:
            return 'stable'
    
    def calculate_knowledge_growth(self):
        """Calculate knowledge growth over time"""
        growth = {
            'mathematical': len(self.math_knowledge),
            'protein_folding': len(self.protein_knowledge),
            'concepts': len(self.concept_knowledge),
            'total': len(self.math_knowledge) + len(self.protein_knowledge) + len(self.concept_knowledge)
        }
        
        return growth
    
    def export_knowledge(self, export_file=None):
        """Export all knowledge to a file"""
        if export_file is None:
            timestamp = int(time.time())
            export_file = f"knowledge_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': time.time(),
            'export_date': datetime.now().isoformat(),
            'knowledge_summary': self.get_knowledge_summary(),
            'mathematical_knowledge': self.math_knowledge,
            'protein_knowledge': self.protein_knowledge,
            'concept_knowledge': self.concept_knowledge,
            'training_history': self.training_history,
            'knowledge_index': self.knowledge_index
        }
        
        try:
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📤 Knowledge exported to: {export_file}")
            return export_file
        except Exception as e:
            print(f"❌ Error exporting knowledge: {e}")
            return None
    
    def import_knowledge(self, import_file):
        """Import knowledge from a file"""
        try:
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            # Merge imported knowledge with existing
            if 'mathematical_knowledge' in import_data:
                self.math_knowledge.update(import_data['mathematical_knowledge'])
            
            if 'protein_knowledge' in import_data:
                self.protein_knowledge.update(import_data['protein_knowledge'])
            
            if 'concept_knowledge' in import_data:
                self.concept_knowledge.update(import_data['concept_knowledge'])
            
            if 'training_history' in import_data:
                self.training_history.extend(import_data['training_history'])
                # Keep only last 100 sessions
                if len(self.training_history) > 100:
                    self.training_history = self.training_history[-100:]
            
            print(f"📥 Knowledge imported from: {import_file}")
            return True
        except Exception as e:
            print(f"❌ Error importing knowledge: {e}")
            return False
    
    def cleanup_old_knowledge(self, max_age_days=30):
        """Clean up old knowledge items"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        cleaned_items = 0
        
        # Clean old mathematical knowledge
        old_math_items = [k for k, v in self.math_knowledge.items() 
                          if current_time - v.get('timestamp', 0) > max_age_seconds]
        for item_id in old_math_items:
            del self.math_knowledge[item_id]
            cleaned_items += 1
        
        # Clean old protein knowledge
        old_protein_items = [k for k, v in self.protein_knowledge.items() 
                            if current_time - v.get('timestamp', 0) > max_age_seconds]
        for item_id in old_protein_items:
            del self.protein_knowledge[item_id]
            cleaned_items += 1
        
        # Clean old concept knowledge
        old_concept_items = [k for k, v in self.concept_knowledge.items() 
                            if current_time - v.get('timestamp', 0) > max_age_seconds]
        for item_id in old_concept_items:
            del self.concept_knowledge[item_id]
            cleaned_items += 1
        
        print(f"🧹 Cleaned {cleaned_items} old knowledge items")
        return cleaned_items

def main():
    """Main function to demonstrate persistent knowledge system"""
    print("🧠 PERSISTENT KNOWLEDGE SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Initialize persistent knowledge system
    knowledge_system = PersistentKnowledgeSystem()
    
    # Show current knowledge summary
    summary = knowledge_system.get_knowledge_summary()
    print(f"\n📊 Current Knowledge Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Add some sample knowledge
    print(f"\n📚 Adding Sample Knowledge...")
    
    # Add mathematical knowledge
    math_id = knowledge_system.add_mathematical_knowledge(
        "What is the derivative of x²?",
        "2x",
        "Using the power rule: d/dx(x^n) = nx^(n-1), so d/dx(x²) = 2x",
        "calculus"
    )
    
    # Add protein knowledge
    protein_id = knowledge_system.add_protein_knowledge(
        "ACDEFG",
        {"helix": 45.2, "sheet": 30.1, "coil": 24.7},
        -25.3,
        {"hydropathy": -0.5, "dominant_structure": "helix"}
    )
    
    # Add concept knowledge
    concept_id = knowledge_system.add_concept_knowledge(
        "Protein Folding",
        "The physical process by which a protein chain acquires its native 3D structure",
        ["Hemoglobin folding", "Enzyme active sites", "Protein misfolding diseases"],
        "biochemistry"
    )
    
    # Record training session
    knowledge_system.record_training_session(
        "mathematical_training",
        [{"problem": "2x = 10", "solution": "x = 5"}],
        95.0,
        300
    )
    
    # Save all knowledge
    knowledge_system.save_all_knowledge()
    
    # Search for knowledge
    print(f"\n🔍 Searching for 'derivative'...")
    results = knowledge_system.search_knowledge("derivative")
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  {result['type']}: {result['data']['problem'][:50]}...")
    
    # Show learning progress
    progress = knowledge_system.get_learning_progress()
    print(f"\n📈 Learning Progress:")
    print(f"  Total Sessions: {progress['total_sessions']}")
    print(f"  Average Accuracy: {progress['average_accuracy']:.1f}%")
    print(f"  Trend: {progress['improvement_trend']}")
    
    # Export knowledge
    export_file = knowledge_system.export_knowledge()
    
    print(f"\n🎉 PERSISTENT KNOWLEDGE SYSTEM DEMONSTRATION COMPLETE!")
    print(f"🧠 All knowledge is now persisted and will be loaded on next startup")
    print(f"📚 The AI will remember all previous training and build upon it")
    print(f"🚀 Ready for continuous learning without starting from scratch!")

if __name__ == "__main__":
    main()
