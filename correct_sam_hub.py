#!/usr/bin/env python3
"""
Correct SAM Web Hub - Head Model + Submodels
Each SAM submodel does: research → self-RAG → augmentation
"""

from flask import Flask, render_template_string, jsonify, request
import time
import threading
from datetime import datetime
from collections import deque
import json
import os
import subprocess
import re
import argparse
import queue
import requests

class CorrectSAMHub:
    def __init__(self):
        self.app = Flask(__name__)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # SAM Head Model - The master model that all submodels clone from
        self.sam_head = None
        self.sam_head_weights_path = 'SAM_STATE/sam_head_weights.bin'
        
        # Submodels are CLONES of SAM head, specialized by context
        # Each submodel is a clone that's been adapted via transfusion/distillation
        self.sam_submodels = {}  # Clone-based submodels
        self.submodel_specializations = {}  # Track how each clone was specialized
        
        # Context-to-submodel mapping (deploy appropriate clone based on context)
        self.context_submodel_map = {
            'conversation': 'sam_conversation',  # Default chat submodel
            'game': 'sam_gameplayer',            # Gaming clone
            'code': 'sam_coder',                 # Coding clone
            'research': 'sam_researcher',        # Deep research clone
        }
        
        # External agents and capabilities
        self.external_agents = {}
        self.ollama_available = False
        self.huggingface_available = False
        
        # Conversation and knowledge
        self.conversation_history = deque(maxlen=200)
        self.knowledge_base = []
        self.pending_knowledge_verification = []  # Queue for verification before saving
        
        # NEW: Search → Augment → Relay → Verify → Save pipeline
        self.search_enabled = True
        self.verification_enabled = True
        self.min_verification_score = 0.7  # Threshold for knowledge acceptance
        
        # Training with full-context batch learning
        self.training_enabled = True
        self.training_queue = queue.Queue()
        self.training_stats = {
            'queued': 0,
            'processed': 0,
            'last_ts': None,
            'last_status': 'idle',
            'verified_correct': 0,
            'verified_incorrect': 0
        }
        
        # Full-context batch for gradient computation
        self.full_context_batch = []  # Accumulate verified examples
        self.batch_size = 5  # Update weights after N verified examples
        
        # MUZE model integration
        self.muze_cli_path = os.getenv('MUZE_CLI_PATH', './muze_chat_cli')
        self.muze_model_path = os.getenv('MUZE_MODEL_PATH', 'MUZE_STATE/muze_enhanced.bin')
        self.muze_last = {
            'ok': None,
            'training_step': None,
            'loss': None,
            'value': None,
            'action': None,
            'ts': None,
            'rc': None,
            'stderr': None,
        }
        
        # Validation and entropy critic
        self.validation_stats = {
            'train_examples': 0,
            'test_examples': 0,
            'validation_cycle': 0,
            'last_validation_ts': None,
            'online_accuracy': [],
            'entropy_scores': [],
            'student_teacher_agreement': []
        }
        self.entropy_critic_enabled = True
        
        # State management
        self.running = True
        self.typing_agent = None
        self.agent_state = {}
        self.muted_agents = set()
        self.msg_seq = 0
        self.last_processed_user_ts = None
        self.external_lock = threading.Lock()
        self.last_external_used = []
        self.last_distilled_ts = None
        
        # Auto-conversation
        self.auto_conversation_active = False
        self.last_auto_seen_msg_id = 0
        self.auto_turn_index = 0
        
        # Latent-space morphogenesis (concept birth) system
        self.morphogenesis_enabled = True
        self.concept_registry = {}  # Track born concepts
        self.concept_birth_threshold = 0.15  # Error threshold for trigger
        self.error_history = []
        self.max_error_history = 100
        
        # Neural core for morphogenesis (connects to C implementation)
        self._neural_core = None
        self._init_neural_core()
        
        # Google Drive backup settings
        self.gdrive_enabled = os.getenv('GDRIVE_FOLDER_ID') is not None
        self.gdrive_folder_id = os.getenv('GDRIVE_FOLDER_ID', '')
        self.last_cloud_backup_ts = None
        
        # Teaching mode
        self.teach_sam_enabled = True
        
        # Initialize
        self.init_sam_system()
        self.setup_routes()
        self._start_training_worker()

    def _start_training_worker(self):
        def worker():
            while self.running:
                try:
                    item = self.training_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    self.training_stats['last_status'] = 'training'
                    self._set_agent_state('sam_head', 'training')
                    self._process_training_item(item)
                    self.training_stats['processed'] += 1
                    self.training_stats['last_ts'] = time.time()
                    self.training_stats['last_status'] = 'idle'
                    self._set_agent_state('sam_head', 'monitoring')
                except Exception as e:
                    self.training_stats['last_status'] = f'error: {e}'
                    self._set_agent_error('sam_head', f'training error: {str(e)}')
                finally:
                    try:
                        self.training_queue.task_done()
                    except Exception:
                        pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _init_neural_core(self):
        """Initialize neural core for morphogenesis and network operations"""
        try:
            from sam_neural_core import SAMNeuralCore
            self._neural_core = SAMNeuralCore()
            
            # Initialize morphogenesis system
            if self._neural_core.initialize_morphogenesis(initial_dim=64, max_dim=256):
                print("🧬 Morphogenesis system initialized via C core")
            else:
                print("⚠️ Morphogenesis init failed, using Python fallback")
                
        except Exception as e:
            print(f"⚠️ Neural core not available: {e}")
            print("   Morphogenesis will use Python implementation")
            self._neural_core = None

    def _enqueue_training(self, user_text, deployed_submodel_id, sam_reply, external_payloads):
        if not self.training_enabled or not self.teach_sam_enabled:
            return
        self.training_stats['queued'] += 1
        self.training_queue.put({
            'user': user_text,
            'submodel_id': deployed_submodel_id,
            'sam_reply': sam_reply,
            'external': external_payloads,
            'transcript': self._format_transcript(limit=16),
            'ts': time.time()
        })

    def _verify_knowledge_accuracy(self, claim, web_evidence):
        """Verify if a knowledge claim is accurate based on web evidence"""
        if not self.ollama_available or not web_evidence:
            return 0.5  # Neutral if can't verify
        
        prompt = (
            "You are a fact-checker. Evaluate if the following claim is supported by the evidence.\n\n"
            f"Claim: {claim}\n\n"
            f"Evidence from web search:\n{web_evidence[:1000]}\n\n"
            "Rate the accuracy (0-10) where:\n"
            "10 = Fully supported by evidence\n"
            "5 = Partially supported or unclear\n"
            "0 = Contradicted by evidence\n\n"
            "Respond with ONLY a number 0-10."
        )
        
        try:
            result = subprocess.run(['ollama', 'run', 'deepseek-r1', prompt], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout.strip()
                # Extract number
                for word in output.split():
                    try:
                        score = float(word)
                        if 0 <= score <= 10:
                            return score / 10.0  # Normalize to 0-1
                    except:
                        continue
        except Exception:
            pass
        
        return 0.5  # Default neutral

    def _generate_training_qa_from_search(self, user_query: str, search_results: str, num_questions: int = 3) -> list:
        """Generate synthetic Q&A pairs from web search results for training
        
        Args:
            user_query: Original user question
            search_results: Web search results text
            num_questions: Number of Q&A pairs to generate
            
        Returns:
            List of (question, answer) tuples
        """
        if not self.ollama_available:
            return []
        
        # Chunk search results if too long
        max_chars = 2000
        excerpt = search_results[:max_chars]
        
        prompt = f"""Based on the user's question and the search results below, generate {num_questions} factual question-answer pairs that could be used to train an AI assistant. The Q&A should be directly answerable from the search results.

User's original question: {user_query}

Search results:
{excerpt}

Generate exactly {num_questions} Q&A pairs in this format:
Q1: [question]
A1: [answer from search results]

Q2: [question]
A2: [answer from search results]

Q3: [question]
A3: [answer from search results]"""

        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama2', prompt],
                capture_output=True,
                text=True,
                timeout=30  # Short timeout for this task
            )
            
            if result.returncode != 0:
                return []
            
            # Parse Q&A pairs
            qa_pairs = []
            output = result.stdout.strip()
            
            # Look for Q1/Q2/Q3 pattern
            for i in range(1, num_questions + 1):
                q_pattern = f"Q{i}:"
                a_pattern = f"A{i}:"
                
                q_start = output.find(q_pattern)
                a_start = output.find(a_pattern)
                
                if q_start != -1 and a_start != -1:
                    q_end = output.find('\n', q_start) if output.find('\n', q_start) != -1 else len(output)
                    a_end = output.find(f'Q{i+1}:', a_start) if i < num_questions else len(output)
                    if a_end == -1:
                        a_end = len(output)
                    
                    q_text = output[q_start + len(q_pattern):q_end].strip()
                    a_text = output[a_start + len(a_pattern):a_end].strip()
                    
                    if q_text and a_text:
                        qa_pairs.append((q_text, a_text))
            
            return qa_pairs
            
        except subprocess.TimeoutExpired:
            print("⚠️ Training Q&A generation timeout")
            return []
        except Exception as e:
            print(f"⚠️ Training Q&A generation error: {e}")
            return []

    def _save_synthetic_training_qa(self, original_query: str, qa_pairs: list, search_results: str):
        """Save generated Q&A pairs as synthetic training data
        
        Args:
            original_query: Original user question that triggered search
            qa_pairs: List of (question, answer) tuples
            search_results: Source search results
        """
        import time
        import json
        
        timestamp = time.time()
        
        for i, (q, a) in enumerate(qa_pairs):
            # Create conversation format
            conversation = [
                {
                    'id': f'synthetic_web_{int(timestamp)}_{i}_q',
                    'sender': 'You',
                    'message': q,
                    'timestamp': time.strftime('%H:%M:%S'),
                    'ts': timestamp + i * 2,
                    'agent_id': 'user',
                    'color': '#888888'
                },
                {
                    'id': f'synthetic_web_{int(timestamp)}_{i}_a',
                    'sender': 'SAM',
                    'message': a,
                    'timestamp': time.strftime('%H:%M:%S'),
                    'ts': timestamp + i * 2 + 1,
                    'agent_id': 'sam_conversation',
                    'color': '#3498db',
                    'verified': True,
                    'source': 'web_search_synthetic',
                    'original_query': original_query,
                    'search_evidence': search_results[:500]  # Truncated
                }
            ]
            
            # Save to training data directory
            os.makedirs('SAM_TRAINING_DATA', exist_ok=True)
            output_file = f'SAM_TRAINING_DATA/synthetic_web_{int(timestamp)}_{i}.jsonl'
            
            with open(output_file, 'a') as f:
                for turn in conversation:
                    f.write(json.dumps(turn) + '\n')

    def process_with_search_pipeline(self, user_text, submodel_id):
        """
        Search → Generate Training Data → Augment → Relay → Verify → Save pipeline
        This fixes the "random spewing" by grounding responses in verified facts
        and generates synthetic training data for continuous learning
        """
        print(f"🔍 Starting search pipeline for: {user_text[:50]}...")
        
        # Step 1: SEARCH - Get real information from web
        search_results = self._maybe_web_lookup(user_text)
        if not search_results:
            print("⚠️ No search results found, using knowledge base fallback")
            search_results = self._retrieve_relevant_knowledge(user_text, limit=3)
            search_results = "\n".join([str(item) for item in search_results]) if search_results else ""
        
        print(f"📚 Search results obtained: {len(search_results)} chars")
        
        # Step 1.5: GENERATE SYNTHETIC TRAINING DATA from search results
        # This creates Q&A pairs for continuous learning
        if len(search_results) > 200:  # Only if we have substantial content
            print("🎓 Generating synthetic training data from search results...")
            training_qa = self._generate_training_qa_from_search(user_text, search_results)
            if training_qa:
                self._save_synthetic_training_qa(user_text, training_qa, search_results)
                print(f"💾 Saved {len(training_qa)} training Q&A pairs")
        
        # Step 2: AUGMENT - Neural network processes search results
        augmented_response = self._augment_with_neural_net(user_text, search_results, submodel_id)
        print(f"🧠 Neural augmentation complete")
        
        # Step 3: RELAY - Summarize and reformulate for user
        relayed_response = self._relay_to_user(user_text, augmented_response, submodel_id)
        print(f"💬 Relayed response prepared")
        
        # Step 4: VERIFY - Check accuracy before showing to user
        if self.verification_enabled:
            verification_score = self._verify_knowledge_accuracy(relayed_response, search_results)
            print(f"✓ Verification score: {verification_score:.2f}/1.0")
            
            if verification_score < self.min_verification_score:
                # If not verified, mark for human review or use fallback
                print(f"⚠️ Low verification score ({verification_score:.2f}), adding disclaimer")
                relayed_response = f"[Note: This response has low confidence ({verification_score:.0%})]\n\n{relayed_response}"
                
                # Don't save low-confidence knowledge
                return relayed_response, False, verification_score
        
        # Step 5: SAVE - If verified, save to knowledge base
        is_verified = True
        if self.verification_enabled:
            is_verified = verification_score >= self.min_verification_score
            
        if is_verified:
            self._save_verified_knowledge(user_text, relayed_response, search_results, verification_score if self.verification_enabled else 0.8)
            print(f"💾 Verified knowledge saved")
        
        return relayed_response, is_verified, verification_score if self.verification_enabled else 0.8

    def _augment_with_neural_net(self, user_query, search_results, submodel_id):
        """Use SAM to process and understand search results"""
        submodel = self.sam_submodels.get(submodel_id, {})
        specialty = submodel.get('specialty', 'General knowledge')
        
        # Use Ollama to process the search results through SAM's "perspective"
        if self.ollama_available:
            prompt = (
                f"You are {submodel.get('name', 'SAM')}, {submodel.get('identifier', 'Assistant')}. "
                f"Your specialty: {specialty}.\n\n"
                f"User question: {user_query}\n\n"
                f"Search results:\n{search_results[:1500]}\n\n"
                f"Synthesize these search results into a clear, accurate answer. "
                f"Focus on facts from the search results. Do not hallucinate. "
                f"Keep your response factual and grounded in the provided evidence."
            )
            
            try:
                cmd = ['ollama', 'run', 'llama2', prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                if result.returncode == 0:
                    return self._sanitize_llm_text(result.stdout)
            except subprocess.TimeoutExpired:
                print(f"⚠️ Augmentation timeout, using fallback")
                return None
            except Exception as e:
                print(f"Augmentation error: {e}")
        
        # Fallback: extract key sentences from search results
        sentences = search_results.split('.')[:3]
        return '. '.join(sentences) + '.'

    def _relay_to_user(self, user_query, augmented_response, submodel_id):
        """Reformulate the augmented response for the user"""
        submodel = self.sam_submodels.get(submodel_id, {})
        
        # Make it conversational based on submodel personality
        if self.ollama_available:
            prompt = (
                f"You are {submodel.get('name', 'SAM')} ({submodel.get('identifier', 'Assistant')}). "
                f"Personality: {submodel.get('personality', 'helpful')}.\n\n"
                f"User asked: {user_query}\n\n"
                f"Factual answer (from verified sources): {augmented_response}\n\n"
                f"Relay this information to the user in a natural, conversational way. "
                f"Be friendly but accurate. Don't make up anything not in the factual answer."
            )
            
            try:
                cmd = ['ollama', 'run', 'llama2', prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                if result.returncode == 0:
                    return self._sanitize_llm_text(result.stdout)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
        
        # Fallback: return the augmented response directly
        return augmented_response

    def _save_verified_knowledge(self, user_query, verified_response, search_evidence, confidence_score):
        """Save verified knowledge to knowledge base"""
        knowledge_item = {
            'kind': 'verified_knowledge',
            'timestamp': time.time(),
            'query': user_query,
            'response': verified_response,
            'evidence': search_evidence[:2000],  # Store truncated evidence
            'confidence': confidence_score,
            'verification_method': 'llm_critic',
            'tags': ['verified', 'web_sourced']
        }
        
        # Add to pending verification queue first
        self.pending_knowledge_verification.append(knowledge_item)
        
        # Also save immediately to knowledge base
        self.save_to_knowledge_base(knowledge_item)
        
        # Add to full-context batch for training
        self.full_context_batch.append({
            'user': user_query,
            'sam_reply': verified_response,
            'evidence': search_evidence,
            'confidence': confidence_score,
            'verified': True
        })

    def _process_full_context_batch(self):
        """Process accumulated verified examples with full-context batch learning"""
        if len(self.full_context_batch) < self.batch_size:
            return  # Not enough examples yet
        
        print(f"📊 Processing full-context batch of {len(self.full_context_batch)} verified examples")
        
        # Compute batch-level statistics for dominant compression
        avg_confidence = sum(item['confidence'] for item in self.full_context_batch) / len(self.full_context_batch)
        verified_count = sum(1 for item in self.full_context_batch if item['verified'])
        
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Verified examples: {verified_count}/{len(self.full_context_batch)}")
        
        # Clear the batch after processing
        self.full_context_batch = []
        
        # Update training stats
        self.training_stats['verified_correct'] += verified_count

    # =================================================================
    # CLONE-BASED SUBMODEL MANAGEMENT
    # Submodels are clones of SAM head, specialized via transfusion/distillation
    # =================================================================
    
    def clone_sam_head(self, clone_id, specialization_context):
        """
        Create a new submodel by cloning SAM head and specializing it
        via transfusion/distillation for a specific context
        
        This now uses the SAM neural core for actual weight cloning.
        """
        print(f"🧬 Cloning SAM head → {clone_id} for context: {specialization_context}")
        
        # Use neural core for actual network operations
        try:
            from sam_neural_core import SAMNetworkManager
            
            # Initialize network manager if not already done
            if not hasattr(self, '_network_manager'):
                # Create a mock core for network management (full core loads C lib)
                self._network_manager = SAMNetworkManager(None)
                # Create base SAM head network if not exists
                if 'sam_head' not in self._network_manager.active_networks:
                    self._network_manager.create_network(
                        'sam_head',
                        input_dim=768,
                        hidden_dims=[512, 256, 128],
                        output_dim=64
                    )
            
            # Clone the network weights
            clone_net = self._network_manager.clone_network(
                'sam_head', 
                clone_id, 
                specialization_context
            )
            
            if clone_net:
                print(f"✓ Neural weights cloned: {clone_net['total_params']} parameters")
                weights_status = 'weights_cloned'
            else:
                weights_status = 'clone_failed'
                
        except Exception as e:
            print(f"⚠️ Neural cloning failed: {e}, using metadata fallback")
            weights_status = 'metadata_only'
        
        # Create full clone record with neural network info
        clone = {
            'id': clone_id,
            'name': 'SAM',
            'identifier': specialization_context.replace('_', ' ').title(),
            'parent': 'sam_head',
            'clone_method': 'transfusion_distillation',
            'specialization': specialization_context,
            'task': f'Handles {specialization_context} tasks',
            'specialty': f'Cloned and specialized for {specialization_context}',
            'personality': 'adapted from head model via context-based distillation',
            'color': self._generate_clone_color(clone_id),
            'capabilities': ['search_augment_relay', 'verify_before_save', 'full_context_learning'],
            'role': f'{specialization_context} Handler',
            'weights_path': f'SAM_STATE/clones/{clone_id}_weights.bin',
            'weights_status': weights_status,
            'created_at': time.time(),
            'neural_config': clone_net if 'clone_net' in dir() else None
        }
        
        self.sam_submodels[clone_id] = clone
        self.submodel_specializations[clone_id] = {
            'context': specialization_context,
            'distillation_steps': 0,
            'verified_examples': 0
        }
        
        # Initialize agent state
        self.agent_state[clone_id] = {
            'present': True,
            'state': 'idle',
            'since': time.time(),
            'name': clone['name'],
            'last_error': None,
            'is_clone': True
        }
        
        print(f"✓ Clone {clone_id} created successfully ({weights_status})")
        return clone
    
    def _generate_clone_color(self, clone_id):
        """Generate a unique color for each clone"""
        colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f',
            '#e67e22', '#1abc9c', '#34495e', '#16a085', '#27ae60'
        ]
        # Use hash of clone_id to pick consistent color
        idx = hash(clone_id) % len(colors)
        return colors[idx]
    
    def deploy_clone_for_context(self, context):
        """
        Deploy the appropriate clone for a given context
        Creates clone on-demand if it doesn't exist
        """
        # Map context to clone ID
        context_lower = context.lower()
        clone_id = None
        
        for ctx_key, ctx_clone_id in self.context_submodel_map.items():
            if ctx_key in context_lower:
                clone_id = ctx_clone_id
                break
        
        if not clone_id:
            clone_id = 'sam_conversation'  # Default
        
        # Create clone if it doesn't exist
        if clone_id not in self.sam_submodels:
            specialization = clone_id.replace('sam_', '')
            self.clone_sam_head(clone_id, specialization)
        
        return clone_id

    def _maybe_web_lookup(self, query):
        """Real web research using DuckDuckGo for all SAM models"""
        q = (query or '').strip()
        if not q:
            return ''
        try:
            # Try to use DDGS if available (better DuckDuckGo integration)
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=3))
                    if results:
                        snippets = [r['body'] for r in results if r.get('body')]
                        return "\n\n".join(snippets)[:800]
            except ImportError:
                pass
            
            # Fallback to direct DuckDuckGo HTML scraping
            url = f"https://duckduckgo.com/html/?q={requests.utils.quote(q)}"
            r = requests.get(url, timeout=8, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if r.status_code != 200:
                return ''
            
            # Better snippet extraction
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            results = soup.find_all('a', class_='result__a')
            snippets = []
            for result in results[:3]:
                text = result.get_text(strip=True)
                if text:
                    snippets.append(text)
            
            return "\n\n".join(snippets)[:800] if snippets else ''
        except Exception as e:
            print(f"Web lookup error: {e}")
            return ''

    def _ollama_teacher_improve(self, user_text, sam_reply, transcript, web_snippet, teacher_responses):
        # Uses DeepSeek as critic/teacher to propose an improved answer.
        if not self.ollama_available:
            return None
        prompt = (
            "You are a strict critic/teacher. Improve the SAM reply to be more helpful and accurate. "
            "Return ONLY the improved reply, no analysis.\n\n"
            f"Conversation:\n{transcript}\n\n"
            f"User message: {user_text}\n\n"
            f"SAM reply: {sam_reply}\n\n"
        )
        if web_snippet:
            prompt += f"Web snippet (may be noisy): {web_snippet}\n\n"
        if teacher_responses:
            joined = "\n".join([f"{t['name']}: {t['text']}" for t in teacher_responses])
            prompt += f"Other agent replies:\n{joined}\n\n"
        prompt += "Improved reply:"

        try:
            result = subprocess.run(['ollama', 'run', 'deepseek-r1', prompt], capture_output=True, text=True, timeout=25)
            if result.returncode == 0:
                return self._sanitize_llm_text(result.stdout)
        except Exception:
            return None
        return None

    def _process_training_item(self, item):
        user_text = item.get('user', '')
        sam_reply = item.get('sam_reply', '')
        transcript = item.get('transcript', '')
        external = item.get('external') or []

        web_snippet = self._maybe_web_lookup(user_text)
        improved = self._ollama_teacher_improve(user_text, sam_reply, transcript, web_snippet, external)

        # Convert teacher signal into a numeric reward (placeholder):
        # if teacher produced an improved reply, reward positive; else small.
        reward = 0.2
        if improved:
            reward = 0.8

        # Call MUZE C model train step (real weights update + checkpoint)
        try:
            os.makedirs(os.path.dirname(self.muze_model_path) or '.', exist_ok=True)
            cli = self.muze_cli_path
            if not os.path.isabs(cli):
                cli = os.path.join(self.base_dir, cli.lstrip('./'))
            model_path = self.muze_model_path
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.base_dir, model_path)

            payload = {
                'text': user_text,
                'train': 1,
                'reward': reward,
                'done': 0,
                'model_path': model_path,
            }
            proc = subprocess.run(
                [cli],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.base_dir,
            )
            out = (proc.stdout or '').strip()
            err = (proc.stderr or '').strip()

            self.muze_last.update({'rc': proc.returncode, 'stderr': err[:300] if err else None})

            if proc.returncode == 0:
                # robust JSON extraction: take first {...}
                start = out.find('{')
                end = out.rfind('}')
                if start != -1 and end != -1 and end > start:
                    j = json.loads(out[start:end+1])
                    self.muze_last.update({
                        'ok': j.get('ok'),
                        'training_step': j.get('training_step'),
                        'loss': j.get('loss'),
                        'value': j.get('value'),
                        'action': j.get('action'),
                        'ts': time.time(),
                    })
                else:
                    self._set_agent_error('sam_head', 'muze_cli: no JSON in stdout')
            else:
                self._set_agent_error('sam_head', f"muze_cli rc={proc.returncode}")
        except Exception as e:
            self._set_agent_error('sam_head', f"muze_cli error: {str(e)}")
        
        # Upload to Google Drive if enabled (async to not block)
        if self.gdrive_enabled and self.muze_last.get('ok') and self.muze_last.get('training_step', 0) % 10 == 0:
            threading.Thread(target=self._upload_to_gdrive, daemon=True).start()

        record = {
            'kind': 'conversation_training',
            'timestamp': item.get('ts', time.time()),
            'user': user_text,
            'sam_reply': sam_reply,
            'improved_reply': improved,
            'transcript': transcript,
            'web_snippet': web_snippet,
            'teacher_replies': external,
            'muze': dict(self.muze_last),
        }

        # Persist as training data
        os.makedirs('SAM_TRAINING_DATA', exist_ok=True)
        fname = f"SAM_TRAINING_DATA/turn_{int(time.time()*1000)}.json"
        with open(fname, 'w') as f:
            json.dump(record, f, indent=2)

        # Also distill to knowledge base (keeps self-rag improving)
        distill = {
            'kind': 'distillation',
            'submodel': self.sam_submodels.get(item.get('submodel_id'), {}).get('name', item.get('submodel_id')),
            'user': user_text,
            'summary': f"User: {user_text}\nSAM: {sam_reply}\nImproved: {improved or ''}"[:1200],
            'timestamp': time.time()
        }
        self.save_to_knowledge_base(distill)
        
        # Update concept utilities with current error
        current_error = self.muze_last.get('loss', 0.5) or 0.5
        self.update_concept_utilities(current_error)
        
        # Compute consciousness loss (new AGI component)
        try:
            from consciousness_loss import ConsciousnessLossModule
            if not hasattr(self, '_consciousness_module'):
                self._consciousness_module = ConsciousnessLossModule(latent_dim=64, action_dim=16)
            
            # Create state for consciousness computation
            import numpy as np
            z_t = np.random.randn(64)  # Current latent state
            a_t = np.random.randn(16)  # Action taken
            m_t = np.random.randn(64)  # Memory/context
            z_next = np.random.randn(64)  # Next state
            
            # Use the compute_loss method directly instead of ConsciousnessState
            try:
                # Create dummy reward for consciousness computation
                reward = np.random.randn(1)
                losses = self._consciousness_module.compute_loss(
                    torch.tensor(z_t, dtype=torch.float32),
                    torch.tensor(a_t, dtype=torch.float32),
                    torch.tensor(z_next, dtype=torch.float32),
                    torch.tensor(m_t, dtype=torch.float32),
                    torch.tensor(reward, dtype=torch.float32),
                    10000  # dummy num_params
                )
                
                # Log consciousness metrics
                print(f"🧠 Consciousness Score: {losses['consciousness_score']:.4f}")
                print(f"   L_cons: {losses['l_cons']:.4f}")
                print(f"   Is Conscious: {losses['consciousness_score'] > 0.7}")
                
                # Save to knowledge base
                self.save_to_knowledge_base({
                    'kind': 'consciousness_metrics',
                    'consciousness_score': losses['consciousness_score'],
                    'l_cons': losses['l_cons'],
                    'is_conscious': losses['consciousness_score'] > 0.7,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                print(f"⚠️ Consciousness computation error: {e}")
                # Continue without consciousness metrics
            
        except Exception as e:
            print(f"⚠️ Consciousness loss computation failed: {e}")
        
        # Check for morphogenesis trigger (concept birth)
        if self.check_morphogenesis_trigger(current_error):
            # Birth a new concept based on the context
            context = user_text[:50] if user_text else "general"
            concept_id = self.birth_concept(
                concept_name=f"concept_from_{context.replace(' ', '_')}",
                context_trigger=user_text
            )
            print(f"🧬 New concept born during training: {concept_id}")
        
        # Run entropy critic evaluation if we have teacher feedback
        if improved and self.entropy_critic_enabled:
            critic_scores = self._ollama_entropy_critic(sam_reply, improved, user_text)
            if critic_scores:
                # Store feedback for student-teacher dynamic analysis
                feedback_record = {
                    'timestamp': time.time(),
                    'context': user_text,
                    'student_reply': sam_reply,
                    'teacher_reply': improved,
                    'entropy_score': critic_scores.get('entropy_score', 5.0),
                    'alignment': critic_scores.get('alignment', 5.0),
                    'hallucination_risk': critic_scores.get('hallucination', 5.0),
                    'missing_insights': critic_scores.get('missing', '')
                }
                self.teacher_feedback_buffer.append(feedback_record)
                # Keep only recent feedback
                if len(self.teacher_feedback_buffer) > 100:
                    self.teacher_feedback_buffer = self.teacher_feedback_buffer[-50:]
                
                # Adjust reward based on entropy critic feedback
                alignment = critic_scores.get('alignment', 5.0)
                if alignment >= 7.0:
                    reward = 1.0  # Good alignment - high reward
                elif alignment >= 5.0:
                    reward = 0.6  # Moderate alignment
                else:
                    reward = 0.2  # Poor alignment - low reward
                
                print(f"🎯 Entropy Critic: alignment={alignment:.1f}/10, reward={reward:.2f}")
        
        # Run validation cycle every 10 processed items
        if self.training_stats.get('processed', 0) % 10 == 0:
            self._run_validation_cycle()

    def _upload_to_gdrive(self):
        """Upload MUZE model checkpoint to Google Drive for cloud backup"""
        try:
            # Check if model file exists
            if not os.path.exists(self.muze_model_path):
                return
            
            # Try to use gdrive CLI tool if available
            try:
                result = subprocess.run(
                    ['gdrive', 'upload', '--parent', self.gdrive_folder_id, self.muze_model_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    self.last_cloud_backup_ts = time.time()
                    print(f"☁️  Cloud backup successful: {result.stdout.strip()}")
                else:
                    print(f"⚠️  Cloud backup failed: {result.stderr.strip()}")
            except FileNotFoundError:
                # gdrive CLI not installed, try alternative methods
                # Copy to a sync folder if configured
                sync_folder = os.getenv('GDRIVE_SYNC_FOLDER', '')
                if sync_folder and os.path.isdir(sync_folder):
                    import shutil
                    dest = os.path.join(sync_folder, os.path.basename(self.muze_model_path))
                    shutil.copy2(self.muze_model_path, dest)
                    self.last_cloud_backup_ts = time.time()
                    print(f"☁️  Copied to sync folder: {dest}")
        except Exception as e:
            print(f"⚠️  Cloud backup error: {e}")

    def _get_recent_messages(self, limit=12):
        msgs = list(self.conversation_history)[-limit:]
        return msgs

    def _format_transcript(self, limit=12):
        msgs = self._get_recent_messages(limit=limit)
        lines = []
        for m in msgs:
            sender = m.get('sender', 'Unknown')
            text = (m.get('message') or '').strip().replace('\n', ' ')
            if len(text) > 300:
                text = text[:300] + '...'
            lines.append(f"{sender}: {text}")
        return "\n".join(lines)

    def _iter_independent_sams(self):
        """Iterate over independent SAM models"""
        for agent_id, agent in self.sam_models.items():
            if agent_id in self.muted_agents:
                continue
            yield agent_id, agent
        
    def generate_sam_conversation_response(self, sam_id, last_message, conversation_context):
        """Generate a contextual response from an independent SAM model"""
        sam = self.sam_models.get(sam_id)
        if not sam:
            return None
        
        # Get recent conversation history for context
        recent_msgs = self._get_recent_messages(limit=8)
        conversation_thread = []
        for msg in recent_msgs:
            conversation_thread.append(f"{msg['sender']}: {msg['message']}")
        
        # Build a prompt that emphasizes responding to the specific message
        transcript = "\n".join(conversation_thread[-6:])  # Last 6 messages for context
        
        prompt = (
            f"You are {sam['name']}, a {sam['identity']}. {sam['personality']}.\n"
            f"Your conversational style is: {sam['style']}.\n"
            f"You are participating in a group discussion. Respond naturally to the last message, "
            f"building on what was said, challenging it, or exploring it further based on your personality.\n"
            f"Keep your response short (1-2 sentences) and conversational.\n"
            f"Do NOT explain who you are, just respond naturally.\n\n"
            f"Recent conversation:\n{transcript}\n\n"
            f"Last message to respond to: {last_message}\n\n"
            f"Your response:"
        )
        
        try:
            # Use Ollama if available, otherwise generate a contextual response
            if self.ollama_available:
                cmd = ['ollama', 'run', 'llama2', prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    return self._sanitize_llm_text(result.stdout)
        except Exception:
            pass
        
        # Fallback: generate contextual response based on personality
        return self._generate_fallback_response(sam, last_message)
    
    def _generate_fallback_response(self, sam, last_message):
        """Generate a contextual fallback response based on SAM's personality"""
        style = sam.get('style', '')
        
        # Extract key topic from last message
        words = last_message.split()[:5]
        topic = ' '.join(words) if words else 'that'
        
        if 'question' in style:
            return f"That's interesting about {topic}. What makes you think that?"
        elif 'practical' in style:
            return f"How does {topic} apply in practice? I'm curious about the implementation."
        elif 'critical' in style:
            return f"Wait, have we considered the edge cases with {topic}?"
        elif 'creative' in style:
            return f"Building on {topic} - what if we took this in a completely different direction?"
        else:
            return f"Interesting point about {topic}. Tell me more."
    
    def verify_conversation_coherence(self, messages):
        """Use LLM to verify conversation is coherent and on-topic"""
        if not self.ollama_available or len(messages) < 2:
            return True, None
        
        transcript = "\n".join([f"{m['sender']}: {m['message']}" for m in messages[-4:]])
        
        prompt = (
            f"Analyze this conversation excerpt and determine if it's coherent and on-topic:\n\n"
            f"{transcript}\n\n"
            f"Respond with ONLY 'COHERENT' if the conversation flows logically and stays on topic, "
            f"or 'INCOHERENT: [brief reason]' if it's random, off-topic, or nonsensical."
        )
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                response = result.stdout.strip().upper()
                if 'INCOHERENT' in response:
                    return False, response
                return True, None
        except Exception:
            pass
        
        return True, None
    
    def _ollama_entropy_critic(self, student_reply, teacher_reply, context):
        """Use LLM as entropy critic to evaluate student-teacher alignment"""
        if not self.ollama_available or not self.entropy_critic_enabled:
            return None
        
        prompt = (
            "You are an entropy critic evaluating how well a student model's response "
            "aligns with an expert teacher's improved version.\n\n"
            f"Context: {context}\n\n"
            f"Student (SAM) reply: {student_reply}\n\n"
            f"Teacher (expert) improved reply: {teacher_reply}\n\n"
            "Rate the student's response on:\n"
            "1. INFORMATION_ENTROPY (0-10): How much useful information was preserved\n"
            "2. CONCEPTUAL_ALIGNMENT (0-10): How well concepts match the teacher\n"
            "3. MISSING_INSIGHTS: What key points did the student miss?\n"
            "4. HALLUCINATION_RISK (0-10): Did the student add false info?\n\n"
            "Format: ENTROPY_SCORE=X|ALIGNMENT=Y|MISSING=brief|HALLUCINATION=Z"
        )
        
        try:
            result = subprocess.run(['ollama', 'run', 'deepseek-r1', prompt], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                output = result.stdout.strip()
                # Parse scores from output
                scores = {}
                for part in output.split('|'):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        key = key.strip().upper()
                        val = val.strip()
                        if key in ['ENTROPY_SCORE', 'ALIGNMENT', 'HALLUCINATION']:
                            try:
                                scores[key.lower()] = float(val)
                            except:
                                scores[key.lower()] = 5.0
                        else:
                            scores[key.lower()] = val
                return scores
        except Exception as e:
            print(f"Entropy critic error: {e}")
        return None
    
    def _run_validation_cycle(self):
        """Run real-time online validation of training examples"""
        processed = self.training_stats.get('processed', 0)
        if processed < 5:
            return  # Not enough data yet
        
        self.validation_stats['validation_cycle'] += 1
        cycle = self.validation_stats['validation_cycle']
        
        # Get recent training examples
        train_dir = 'SAM_TRAINING_DATA'
        if not os.path.exists(train_dir):
            return
        
        files = sorted([f for f in os.listdir(train_dir) if f.endswith('.json')])[-10:]
        if len(files) < 5:
            return
        
        # Split: use 80% for training validation, 20% as holdout test
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        scores = []
        agreements = []
        
        for fname in train_files:
            try:
                with open(os.path.join(train_dir, fname), 'r') as f:
                    data = json.load(f)
                
                sam_reply = data.get('sam_reply', '')
                improved = data.get('improved_reply', '')
                user = data.get('user', '')
                
                if sam_reply and improved:
                    # Run entropy critic
                    critic_scores = self._ollama_entropy_critic(sam_reply, improved, user)
                    if critic_scores:
                        scores.append(critic_scores.get('alignment', 5.0))
                        
                        # Check student-teacher agreement (similarity heuristic)
                        sam_words = set(sam_reply.lower().split())
                        improved_words = set(improved.lower().split())
                        if sam_words and improved_words:
                            overlap = len(sam_words & improved_words) / len(sam_words | improved_words)
                            agreements.append(overlap)
            except Exception as e:
                continue
        
        # Update validation stats
        if scores:
            avg_score = sum(scores) / len(scores)
            self.validation_stats['entropy_scores'].append(avg_score)
            self.validation_stats['last_validation_ts'] = time.time()
            
            print(f"📊 Validation Cycle #{cycle}: Avg Entropy Score={avg_score:.2f}/10")
        
        if agreements:
            avg_agreement = sum(agreements) / len(agreements)
            self.validation_stats['student_teacher_agreement'].append(avg_agreement)
            print(f"🤝 Student-Teacher Agreement: {avg_agreement:.1%}")
        
        # Online accuracy (rolling window)
        self.validation_stats['online_accuracy'] = scores[-20:]
    
    def get_validation_metrics(self):
        """Get current validation metrics for UI display"""
        metrics = {
            'cycle': self.validation_stats['validation_cycle'],
            'train_count': self.training_stats.get('processed', 0),
            'entropy_score': None,
            'agreement': None,
            'trend': 'stable'
        }
        
        if self.validation_stats['entropy_scores']:
            metrics['entropy_score'] = self.validation_stats['entropy_scores'][-1]
            # Calculate trend
            if len(self.validation_stats['entropy_scores']) >= 3:
                recent = self.validation_stats['entropy_scores'][-3:]
                if recent[-1] > recent[0]:
                    metrics['trend'] = 'improving'
                elif recent[-1] < recent[0]:
                    metrics['trend'] = 'declining'
        
        if self.validation_stats['student_teacher_agreement']:
            metrics['agreement'] = self.validation_stats['student_teacher_agreement'][-1]
        
        return metrics

    def init_sam_system(self):
        """Initialize SAM head model and submodels"""
        
        # SAM Head Model - Only identifies conversations and deploys submodels
        self.sam_head = {
            'name': 'SAM-Head',
            'role': 'Conversation Identifier & Submodel Deployer',
            'function': 'Detect conversation context and deploy appropriate SAM submodel',
            'status': 'monitoring'
        }

        # SAM Submodels - TASK-BASED: Each handles an entire domain/task independently
        # These are for specific tasks/games, not for breaking down parts of a task
        self.sam_submodels = {
            'sam_chessmaster': {
                'name': 'SAM',
                'identifier': 'ChessMaster',
                'task': 'Complete Chess Games',
                'specialty': 'Plays and analyzes complete chess games from start to finish',
                'personality': 'strategic, patient, tactical',
                'color': '#8B4513',  # Chess brown
                'capabilities': ['play_full_games', 'strategy_analysis', 'endgame_expert'],
                'role': 'Chess Game Handler'
            },
            'sam_coder': {
                'name': 'SAM',
                'identifier': 'Coder',
                'task': 'Full Software Development',
                'specialty': 'Writes, debugs, and deploys complete software projects',
                'personality': 'precise, systematic, solution-oriented',
                'color': '#2ecc71',  # Green
                'capabilities': ['fullstack_dev', 'debugging', 'architecture'],
                'role': 'Software Development Handler'
            },
            'sam_researcher': {
                'name': 'SAM',
                'identifier': 'Researcher',
                'task': 'Deep Research Projects',
                'specialty': 'Conducts comprehensive research from start to final report',
                'personality': 'thorough, analytical, evidence-based',
                'color': '#3498db',  # Blue
                'capabilities': ['literature_review', 'data_analysis', 'synthesis'],
                'role': 'Research Project Handler'
            },
            'sam_storyteller': {
                'name': 'SAM',
                'identifier': 'Storyteller',
                'task': 'Complete Story Creation',
                'specialty': 'Creates full narratives from concept to finished story',
                'personality': 'creative, imaginative, structured',
                'color': '#9b59b6',  # Purple
                'capabilities': ['world_building', 'plot_development', 'character_arcs'],
                'role': 'Story Creation Handler'
            }
        }

        self.ollama_available = self._check_ollama_available()
        self.huggingface_available = self._check_huggingface_available()
        self.external_agents = {}

        if self.ollama_available:
            self.external_agents['ollama_deepseek'] = {
                'id': 'ollama_deepseek',
                'name': 'Ollama-DeepSeek',
                'provider': 'ollama',
                'model_name': 'deepseek-r1',
                'specialty': 'Technical Analysis & Reasoning',
                'personality': 'technical, precise, analytical',
                'color': '#1abc9c'
            }
            self.external_agents['ollama_llama2'] = {
                'id': 'ollama_llama2',
                'name': 'Ollama-Llama2',
                'provider': 'ollama',
                'model_name': 'llama2',
                'specialty': 'General Conversation',
                'personality': 'balanced, knowledgeable, conversational',
                'color': '#f1c40f'
            }
            self.external_agents['ollama_mistral'] = {
                'id': 'ollama_mistral',
                'name': 'Ollama-Mistral',
                'provider': 'ollama',
                'model_name': 'mistral',
                'specialty': 'Efficient Language Processing',
                'personality': 'efficient, direct, practical',
                'color': '#3498db'
            }
            self.external_agents['ollama_codellama'] = {
                'id': 'ollama_codellama',
                'name': 'Ollama-CodeLlama',
                'provider': 'ollama',
                'model_name': 'codellama',
                'specialty': 'Code & Technical Reasoning',
                'personality': 'technical, code-focused, precise',
                'color': '#e74c3c'
            }
            self.external_agents['ollama_phi'] = {
                'id': 'ollama_phi',
                'name': 'Ollama-Phi',
                'provider': 'ollama',
                'model_name': 'phi',
                'specialty': 'Compact Efficient Reasoning',
                'personality': 'concise, focused, efficient',
                'color': '#9b59b6'
            }

        if self.huggingface_available:
            self.external_agents['hf_distilgpt2'] = {
                'id': 'hf_distilgpt2',
                'name': 'HF-DistilGPT2',
                'provider': 'huggingface_local',
                'model_name': 'distilgpt2',
                'specialty': 'Fast Local Text Generation',
                'personality': 'concise, helpful',
                'color': '#e74c3c'
            }

        self.external_health = {}
        for agent_id in self.external_agents.keys():
            self.external_health[agent_id] = {
                'consecutive_timeouts': 0,
                'disabled_until': 0.0
            }

        # Initialize agent state (presence + activity)
        self.agent_state = {}
        now = time.time()
        
        # Initialize Independent SAM Models (for conversation)
        self.sam_models = {
            'sam_alpha': {
                'name': 'SAM-Alpha',
                'identity': 'Curious Investigator',
                'personality': 'Asks probing questions, loves to dig deeper into topics',
                'style': 'exploratory, questioning',
                'role': 'Independent Thinker',
                'color': '#e74c3c'
            },
            'sam_beta': {
                'name': 'SAM-Beta',
                'identity': 'Pragmatic Synthesizer',
                'personality': 'Connects dots, finds practical applications and patterns',
                'style': 'synthesizing, practical',
                'role': 'Independent Thinker',
                'color': '#3498db'
            },
            'sam_gamma': {
                'name': 'SAM-Gamma',
                'identity': "Devil's Advocate",
                'personality': 'Challenges assumptions, finds edge cases and counterarguments',
                'style': 'critical, challenging',
                'role': 'Independent Thinker',
                'color': '#9b59b6'
            },
            'sam_delta': {
                'name': 'SAM-Delta',
                'identity': 'Creative Builder',
                'personality': 'Imagines possibilities, builds on ideas creatively',
                'style': 'creative, constructive',
                'role': 'Independent Thinker',
                'color': '#2ecc71'
            }
        }
        
        for agent_id in self.sam_models.keys():
            self.agent_state[agent_id] = {
                'present': True,
                'state': 'idle',
                'since': now,
                'name': self.sam_models[agent_id]['name'],
                'last_error': None
            }
        
        for agent_id in self.sam_submodels.keys():
            self.agent_state[agent_id] = {
                'present': True,
                'state': 'idle',
                'since': now,
                'name': self.sam_submodels[agent_id]['name'],
                'last_error': None
            }
        for agent_id in self.external_agents.keys():
            self.agent_state[agent_id] = {
                'present': True,
                'state': 'idle',
                'since': now,
                'name': self.external_agents[agent_id]['name'],
                'last_error': None
            }
        self.agent_state['sam_head'] = {
            'present': True,
            'state': 'monitoring',
            'since': now,
            'name': 'SAM-Head',
            'last_error': None
        }

    def _set_agent_state(self, agent_id, state):
        now = time.time()
        if agent_id not in self.agent_state:
            self.agent_state[agent_id] = {'present': True, 'state': state, 'since': now, 'name': agent_id, 'last_error': None}
            return
        self.agent_state[agent_id]['present'] = True
        self.agent_state[agent_id]['state'] = state
        self.agent_state[agent_id]['since'] = now

    def _set_agent_error(self, agent_id, err):
        if agent_id not in self.agent_state:
            self._set_agent_state(agent_id, 'idle')
        self.agent_state[agent_id]['last_error'] = (err or '')[:240]

    def _sanitize_llm_text(self, text):
        if not text:
            return text

        t = text.strip()

        # Remove common chain-of-thought style prefixes if the model ignores instructions.
        cot_markers = [
            'thinking...',
            'okay,',
            'alright,',
            'here is my thinking',
            "let's think",
            'analysis:',
            'reasoning:',
            'scratchpad:'
        ]

        lines = [ln.rstrip() for ln in t.splitlines()]
        cleaned = []
        skipping = True
        for ln in lines:
            low = ln.strip().lower()
            if skipping:
                if not low:
                    continue
                if any(low.startswith(m) for m in cot_markers):
                    continue
                # If the first non-empty line looks like internal meta reasoning, skip it.
                if any(p in low for p in ['the user is asking', 'i should', 'my plan', 'step-by-step']):
                    continue
                skipping = False
            cleaned.append(ln)

        t2 = "\n".join(cleaned).strip()
        t2 = t2 if t2 else t

        # Prefer the first paragraph.
        para = t2.split("\n\n")[0].strip()

        # If the paragraph still looks like meta-commentary, try to salvage a short user-facing line.
        meta_hints = [
            'they\'re clearly',
            'the user is',
            'interaction mode',
            'low-stakes',
            'i\'m thinking',
            'i should',
            'plan:',
            'step-by-step',
            'reasoning',
        ]
        low_para = para.lower()
        if any(h in low_para for h in meta_hints) or len(para) > 280:
            # Remove lines that look like analysis/meta.
            keep = []
            for ln in para.splitlines():
                l = ln.strip()
                ll = l.lower()
                if not l:
                    continue
                if any(h in ll for h in meta_hints):
                    continue
                if ll.startswith(('hmm', 'okay', 'alright')):
                    continue
                keep.append(l)

            candidate = " ".join(keep).strip()
            if not candidate:
                # Fall back to the last line of the original text.
                candidate = (t2.splitlines()[-1] if t2.splitlines() else t2).strip()
            para = candidate

        # Truncate for groupchat feel.
        if len(para) > 300:
            para = para[:300].rstrip() + '...'
        return para

    def _check_ollama_available(self):
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False

    def _check_huggingface_available(self):
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _load_knowledge_items(self):
        knowledge_file = 'KNOWLEDGE_BASE/sam_knowledge.json'
        if not os.path.exists(knowledge_file):
            return []
        try:
            with open(knowledge_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except Exception:
            return []

    def _retrieve_relevant_knowledge(self, query, limit=3):
        items = self._load_knowledge_items()
        if not items:
            return []

        words = [w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(w) >= 4]
        if not words:
            return []

        scored = []
        for it in items:
            blob = json.dumps(it).lower()
            score = sum(1 for w in set(words) if w in blob)
            if score > 0:
                scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def _distill_learning_item(self, user_text, deployed_submodel_id, sam_response, external_responses):
        summary_bits = []
        if user_text:
            summary_bits.append(f"User asked: {user_text[:240]}")
        if sam_response:
            summary_bits.append(f"SAM replied: {sam_response[:240]}")
        for er in external_responses:
            summary_bits.append(f"{er['name']} replied: {er['text'][:240]}")
        return {
            'kind': 'distillation',
            'submodel': self.sam_submodels.get(deployed_submodel_id, {}).get('name', deployed_submodel_id),
            'user': user_text,
            'summary': "\n".join(summary_bits)[:1200],
            'timestamp': time.time()
        }
    
    def deploy_submodel(self, context):
        """
        SAM Head Model deploys appropriate clone-based submodel
        Clones are created on-demand from SAM head and specialized via transfusion
        """
        # Use the clone deployment system
        return self.deploy_clone_for_context(context)
    
    def process_with_submodel(self, submodel_id, context):
        """SAM submodel processes with all three capabilities"""
        submodel = self.sam_submodels[submodel_id]

        retrieved = self._retrieve_relevant_knowledge(context, limit=3)
        retrieved_text = ""
        if retrieved:
            retrieved_text = "\n".join([
                it.get('summary') or it.get('augmentation') or it.get('rag') or str(it) for it in retrieved
            ])
        
        # Internal pipeline (kept for learning/distillation; not shown to user)
        research_results = self.perform_research(context)
        rag_results = self.perform_self_rag(context, research_results, retrieved_text=retrieved_text)
        augmentation_results = self.perform_augmentation(context, research_results, rag_results)
        
        # Save to knowledge base
        self.save_to_knowledge_base({
            'submodel': submodel['name'],
            'context': context,
            'research': research_results,
            'rag': rag_results,
            'augmentation': augmentation_results,
            'timestamp': time.time()
        })
        
        transcript = self._format_transcript(limit=10)
        return self._sam_chat_reply(submodel_id=submodel_id, user_text=context, retrieved=retrieved, transcript=transcript)

    def _sam_chat_reply(self, submodel_id, user_text, retrieved, transcript=""):
        sub = self.sam_submodels.get(submodel_id, {})
        specialty = sub.get('specialty', '')

        hint = ""
        if retrieved:
            # Use retrieved knowledge as a private hint (do not paste raw memory into chat)
            best = retrieved[0]
            hint = best.get('summary') or best.get('augmentation') or best.get('rag') or ""
            hint = (hint or "").strip().replace('\n', ' ')
            if len(hint) > 180:
                hint = hint[:180] + '...'

        txt = (user_text or '').strip()
        low = txt.lower()

        last_sender = None
        last_text = None
        recent = self._get_recent_messages(limit=2)
        if recent:
            last_sender = recent[-1].get('sender')
            last_text = (recent[-1].get('message') or '').strip()

        if any(g in low for g in ['hi', 'hello', 'hey', 'can you hear me']):
            return "Hey — I’m here. What do you want to dig into?"

        if txt.endswith('?'):
            if 'Technical' in specialty:
                return "I can help with that. What language / stack are you using and what’s the exact error or goal?"
            if 'Scientific' in specialty:
                return "Good question. What part do you want—definition, intuition, or a concrete example?"
            if 'Creative' in specialty:
                return "Fun question. Do you want ideas, critique, or a rewrite?"
            return "Got it. What’s the main constraint you care about most?"

        # Default: acknowledge + nudge forward without echoing the whole message
        if hint:
            return "Got it. I’m going to build on what we’ve discussed—what direction do you want next?"
        return "Got it. What should we focus on next?"
    
    def perform_research(self, query):
        """Research capability"""
        # Simulate research without external dependencies
        research_topics = [
            f"Research on {query}: Key concepts and definitions",
            f"Research on {query}: Current state and developments",
            f"Research on {query}: Expert opinions and analysis"
        ]
        return research_topics[int(time.time()) % len(research_topics)]
    
    def perform_self_rag(self, context, research_results, retrieved_text=""):
        """Self-RAG capability - analyze and retrieve relevant information"""
        # Simulate self-RAG analysis
        rag_analyses = [
            f"Self-RAG Analysis: Identified key patterns in {context}",
            f"Self-RAG Analysis: Retrieved relevant information from research results",
            f"Self-RAG Analysis: Generated insights based on context and research"
        ]
        base = rag_analyses[int(time.time()) % len(rag_analyses)]
        if retrieved_text:
            return base + f"\n\nRetrieved from SAM knowledge:\n{retrieved_text[:600]}"
        return base

    def generate_external_response(self, agent_id, message, transcript=""):
        agent = self.external_agents.get(agent_id)
        if not agent:
            return None

        now = time.time()
        health = self.external_health.get(agent_id)
        if health and health.get('disabled_until', 0) > now:
            self._set_agent_state(agent_id, 'idle')
            return None

        provider = agent.get('provider')
        if provider == 'ollama':
            try:
                prompt = (
                    f"You are {agent.get('name','an assistant')}, a {agent.get('specialty', 'assistant').lower()} specialist "
                    f"with personality {agent.get('personality', '')}.\n"
                    f"Reply as a short group chat message.\n"
                    f"Return ONLY the final answer (no hidden reasoning, no analysis, no scratchpad).\n\n"
                    f"Conversation so far (most recent last):\n{transcript}\n\n"
                    f"Message: {message}\n\nAnswer:"
                )
                cmd = ['ollama', 'run', agent['model_name'], prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    if health:
                        health['consecutive_timeouts'] = 0
                    self._set_agent_error(agent_id, None)
                    return self._sanitize_llm_text(result.stdout)
                stderr = (result.stderr or '').strip()
                if stderr:
                    self._set_agent_error(agent_id, stderr)
                return None
            except subprocess.TimeoutExpired:
                if health is not None:
                    health['consecutive_timeouts'] = int(health.get('consecutive_timeouts', 0)) + 1
                    backoff = min(300, 10 * (2 ** (health['consecutive_timeouts'] - 1)))
                    health['disabled_until'] = time.time() + backoff
                    self._set_agent_state(agent_id, 'cooldown')
                self._set_agent_error(agent_id, 'ollama timeout')
                return None
            except Exception as e:
                self._set_agent_error(agent_id, f"ollama error: {str(e)}")
                return None

        if provider == 'huggingface_local':
            try:
                from transformers import pipeline

                gen = pipeline('text-generation', model=agent['model_name'])
                prompt = f"Conversation:\n{transcript}\n\nUser: {message}\nAssistant:"
                out = gen(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
                if out and isinstance(out, list) and 'generated_text' in out[0]:
                    txt = out[0]['generated_text']
                    if txt.startswith(prompt):
                        txt = txt[len(prompt):]
                    self._set_agent_error(agent_id, None)
                    return self._sanitize_llm_text(txt)
                self._set_agent_error(agent_id, 'huggingface returned no text')
                return None
            except Exception as e:
                self._set_agent_error(agent_id, f"huggingface error: {str(e)}")
                return None

        return None
    
    def perform_augmentation(self, context, research, rag):
        """Augmentation capability - enhance and synthesize knowledge"""
        # Simulate knowledge augmentation
        augmentations = [
            f"Knowledge Augmentation: Synthesized {context} with research findings",
            f"Knowledge Augmentation: Enhanced understanding through self-RAG insights",
            f"Knowledge Augmentation: Created comprehensive knowledge synthesis"
        ]
        return augmentations[int(time.time()) % len(augmentations)]
    
    def save_to_knowledge_base(self, data):
        """Save to knowledge base"""
        try:
            os.makedirs('KNOWLEDGE_BASE', exist_ok=True)
            
            # Load existing knowledge
            knowledge_file = 'KNOWLEDGE_BASE/sam_knowledge.json'
            existing_knowledge = []
            
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r') as f:
                    try:
                        existing_knowledge = json.load(f)
                    except:
                        existing_knowledge = []
            
            # Add new knowledge
            existing_knowledge.append(data)
            
            # Save updated knowledge
            with open(knowledge_file, 'w') as f:
                json.dump(existing_knowledge, f, indent=2)
                
        except Exception as e:
            print(f"Error saving to knowledge base: {e}")
    
    def add_message(self, sender, message, agent_id=None):
        """Add message to conversation"""
        color = '#333333'
        if agent_id in self.sam_submodels:
            color = self.sam_submodels.get(agent_id, {}).get('color', color)
        elif agent_id in self.external_agents:
            color = self.external_agents.get(agent_id, {}).get('color', color)
        elif agent_id == 'sam_learning':
            color = '#95a5a6'

        self.msg_seq += 1
        msg = {
            'id': self.msg_seq,
            'sender': sender,
            'message': message,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'ts': time.time(),
            'agent_id': agent_id,
            'color': color
        }
        self.conversation_history.append(msg)
    
    def start_conversation_system(self):
        """Start SAM conversation system"""
        def conversation_thread():
            time.sleep(3)  # Initial delay
            
            # Wait for user to start conversation
            while self.running and len(self.conversation_history) == 0:
                time.sleep(1)
            
            # Continue conversation
            while self.running:
                if len(self.conversation_history) > 0:
                    # Get last message
                    last_msg = list(self.conversation_history)[-1]
                    
                    # SAM Head Model identifies context and deploys submodel
                    if last_msg['sender'] == 'You':
                        if self.last_processed_user_ts == last_msg['timestamp']:
                            time.sleep(0.5)
                            continue
                        deployed_submodel = self.deploy_submodel(last_msg['message'])

                        self._set_agent_state('sam_head', 'deploying')
                        self._set_agent_state(deployed_submodel, 'thinking')
                        
                        # Show typing indicator
                        self.typing_agent = deployed_submodel
                        time.sleep(2)

                        self._set_agent_state(deployed_submodel, 'typing')
                        
                        # NEW: Use search pipeline for grounded, verified responses
                        response, is_verified, confidence = self.process_with_search_pipeline(
                            last_msg['message'], 
                            deployed_submodel
                        )
                        
                        # Add verification indicator to display name
                        submodel = self.sam_submodels[deployed_submodel]
                        display_name = submodel['name']
                        if submodel.get('identifier'):
                            verified_badge = "✓" if is_verified else "~"
                            display_name = f"{submodel['name']} ({submodel['identifier']}) [{verified_badge}]"
                        
                        self.add_message(display_name, response, deployed_submodel)

                        self._set_agent_state(deployed_submodel, 'idle')
                        self._set_agent_state('sam_head', 'monitoring')

                        external_payloads = []
                        self.last_external_used = []

                        def run_external(ext_id, agent, user_text):
                            self._set_agent_state(ext_id, 'thinking')
                            transcript = self._format_transcript(limit=12)
                            ext_text = self.generate_external_response(ext_id, user_text, transcript=transcript)
                            if ext_text:
                                self._set_agent_state(ext_id, 'typing')
                                with self.external_lock:
                                    external_payloads.append({'id': ext_id, 'name': agent['name'], 'text': ext_text})
                                self.add_message(agent['name'], ext_text, ext_id)
                                with self.external_lock:
                                    self.last_external_used.append(ext_id)
                            self._set_agent_state(ext_id, 'idle')

                        if self.external_agents:
                            for ext_id, agent in self.external_agents.items():
                                t = threading.Thread(target=run_external, args=(ext_id, agent, last_msg['message']), daemon=True)
                                t.start()

                        if self.teach_sam_enabled:
                            distill = self._distill_learning_item(
                                user_text=last_msg['message'],
                                deployed_submodel_id=deployed_submodel,
                                sam_response=response,
                                external_responses=external_payloads,
                            )
                            self.save_to_knowledge_base(distill)
                            self.last_distilled_ts = distill.get('timestamp')

                        # Enqueue real training update using teacher + web augmentation
                        with self.external_lock:
                            ext_copy = list(external_payloads)
                        self._enqueue_training(last_msg['message'], deployed_submodel, response, ext_copy)

                        self.last_processed_user_ts = last_msg['timestamp']
                        
                        self.typing_agent = None

                    # Autonomous SAM-to-SAM conversation with proper threading
                    if self.auto_conversation_active:
                        last_id = last_msg.get('id')
                        if last_id and last_id > self.last_auto_seen_msg_id:
                            self.last_auto_seen_msg_id = last_id
                            
                            # Get independent SAM models (not submodels)
                            sam_ids = [aid for aid, _ in self._iter_independent_sams()]
                            if sam_ids and last_msg.get('sender') != 'You':
                                # Find a SAM that should respond based on the last speaker
                                last_sender = last_msg.get('sender', '')
                                
                                # Pick next SAM (round-robin but avoid immediate self-reply)
                                self.auto_turn_index = (self.auto_turn_index + 1) % len(sam_ids)
                                next_id = sam_ids[self.auto_turn_index]
                                
                                # Skip if same as last speaker
                                sam_info = self.sam_models.get(next_id, {})
                                if sam_info.get('name') == last_sender:
                                    next_id = sam_ids[(self.auto_turn_index + 1) % len(sam_ids)]
                                
                                # Generate contextual response
                                self._set_agent_state(next_id, 'thinking')
                                time.sleep(1.5)
                                self._set_agent_state(next_id, 'typing')
                                
                                # Get the actual message content to respond to
                                last_message_text = last_msg.get('message', '')
                                reply = self.generate_sam_conversation_response(
                                    next_id, 
                                    last_message_text, 
                                    self._format_transcript(limit=8)
                                )
                                
                                # Add message
                                sam_name = self.sam_models[next_id]['name']
                                self.add_message(sam_name, reply, next_id)
                                self._set_agent_state(next_id, 'idle')
                                
                                # Verify conversation coherence
                                is_coherent, reason = self.verify_conversation_coherence(
                                    list(self.conversation_history)[-5:]
                                )
                                if not is_coherent:
                                    print(f"⚠️ Conversation coherence issue: {reason}")
                                
                                # Trigger training for SAM conversation
                                if self.teach_sam_enabled:
                                    with self.external_lock:
                                        self._enqueue_training(
                                            last_message_text, 
                                            next_id, 
                                            reply, 
                                            []
                                        )
                    
                    # Random delay before next check
                    time.sleep(3 + (time.time() % 4))
                else:
                    time.sleep(1)
        
        thread = threading.Thread(target=conversation_thread, daemon=True)
        thread.start()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Correct SAM Hub - Head Model + Submodels</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .status {
            background: rgba(0,0,0,0.2);
            padding: 10px 20px;
            text-align: center;
            backdrop-filter: blur(5px);
            font-size: 1.1em;
        }
        .container {
            flex: 1;
            display: flex;
            padding: 20px;
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .system-panel {
            width: 250px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(5px);
        }
        .head-model, .submodel {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .head-model {
            background: rgba(255,215,0,0.1);
            border-left-color: #ffd700;
        }
        .submodel-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        .submodel-role {
            font-size: 0.8em;
            color: rgba(255,255,255,0.8);
            margin: 4px 0;
        }
        .submodel-capabilities {
            font-size: 0.7em;
            color: rgba(255,255,255,0.6);
        }
        .chat-area {
            flex: 1;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(5px);
            display: flex;
            flex-direction: column;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 15px;
            border-radius: 5px;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            max-height: 400px;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 8px;
            animation: fadeIn 0.3s ease;
            border-left: 4px solid #3498db;
            white-space: pre-wrap;
        }
        .message.from-user {
            margin-left: 60px;
            margin-right: 0;
            background: rgba(52, 152, 219, 0.18);
            border-left-color: rgba(52, 152, 219, 0.9);
        }
        .message.from-agent {
            margin-right: 60px;
            background: rgba(255, 255, 255, 0.08);
        }
        .message .meta {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 6px;
        }
        .message .sender {
            font-weight: 700;
            color: #fff;
        }
        .message .time {
            font-size: 0.8em;
            color: rgba(255,255,255,0.65);
        }
        .message .content {
            line-height: 1.35;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 16px;
        }
        .input-area button {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            background: #3498db;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .input-area button:hover {
            background: #2980b9;
        }
        .input-area input::placeholder {
            color: rgba(255,255,255,0.7);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Correct SAM Hub</h1>
        <p>Head Model + Submodels - Each Submodel Does Research + Self-RAG + Augmentation</p>
    </div>
    
    <div class="status">
        <div id="status">🤖 SAM Head Model monitoring...</div>
        <div id="agent-indicators" style="margin-top:8px; font-size:0.95em; opacity:0.95;"></div>
    </div>
    
    <div class="container">
        <div class="system-panel">
            <h3>🧠 SAM System</h3>
            
            <div class="head-model">
                <div class="submodel-name">👑 SAM-Head Model</div>
                <div class="submodel-role">Conversation Identifier & Submodel Deployer</div>
                <div class="submodel-capabilities">Detects context → Deploys appropriate submodel</div>
            </div>
            
            <h4>🤖 SAM Submodels</h4>
            <div id="submodels-list"></div>

            <h4>🧩 Other Agents (Independent)</h4>
            <div id="external-list"></div>
        </div>
        
        <div class="chat-area">
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="message-input" placeholder="Type your message (SAM Head will deploy appropriate submodel)..." />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        let lastMessageCount = 0;
        let lastSeq = 0;
        let submodels = [];
        let externals = [];
        
        // Initialize
        function init() {
            updateSubmodelsList();
            updateExternalList();
            updateStatus();
            updateAgentIndicators();
            loadMessages();
            setInterval(updateStatus, 1000);
            setInterval(updateAgentIndicators, 750);
            setInterval(checkForNewMessages, 2000);
        }
        
        // Check for new messages
        function checkForNewMessages() {
            fetch('/api/messages')
                .then(response => response.json())
                .then(messages => {
                    messages.sort((a, b) => (a.id || 0) - (b.id || 0));
                    for (const m of messages) {
                        if ((m.id || 0) > lastSeq) {
                            addMessage(m);
                            lastSeq = m.id || lastSeq;
                        }
                    }
                    lastMessageCount = messages.length;
                });
        }

        function updateAgentIndicators() {
            fetch('/api/agent_state')
                .then(response => response.json())
                .then(data => {
                    const el = document.getElementById('agent-indicators');
                    const typing = [];
                    const thinking = [];
                    const errors = [];

                    for (const [id, st] of Object.entries(data)) {
                        if (!st || typeof st !== 'object') continue;
                        if (!st.present) continue;
                        const name = st.name || id;
                        if (st.state === 'typing') typing.push(name);
                        if (st.state === 'thinking' || st.state === 'deploying') thinking.push(name);
                        if (st.last_error) errors.push(`${name}: ${st.last_error}`);
                    }

                    const parts = [];
                    if (typing.length) parts.push(`Typing: ${typing.join(', ')}`);
                    if (thinking.length) parts.push(`Thinking: ${thinking.join(', ')}`);
                    if (!typing.length && !thinking.length) parts.push('All agents idle');
                    if (errors.length) parts.push(`Issues: ${errors.slice(0,2).join(' | ')}`);
                    
                    // Add MUZE training metrics with more detail
                    const training = data.training || {};
                    const muze = training.muze_last || {};
                    if (muze.ok === true && muze.training_step !== null) {
                        const metrics = [`step=${muze.training_step}`, `loss=${(muze.loss || 0).toFixed(4)}`];
                        if (muze.value !== null && muze.value !== undefined) {
                            metrics.push(`V=${(muze.value || 0).toFixed(3)}`);
                        }
                        if (muze.action !== null && muze.action !== undefined) {
                            metrics.push(`A=${muze.action}`);
                        }
                        // Add cloud backup indicator
                        if (training.cloud_backup_ts) {
                            const backupAge = Math.round((Date.now()/1000 - training.cloud_backup_ts) / 60);
                            if (backupAge < 60) {
                                metrics.push(`☁️ ${backupAge}m ago`);
                            }
                        }
                        parts.push(`MUZE: ${metrics.join(' ')}`);
                    } else if (training.queued > 0) {
                        parts.push(`Training: queued=${training.queued} processed=${training.processed}`);
                    }
                    
                    // Add validation metrics
                    const validation = data.validation || {};
                    if (validation.entropy_score !== null && validation.entropy_score !== undefined) {
                        const valMetrics = [`entropy=${validation.entropy_score.toFixed(1)}/10`];
                        if (validation.agreement !== null && validation.agreement !== undefined) {
                            valMetrics.push(`agree=${(validation.agreement * 100).toFixed(0)}%`);
                        }
                        valMetrics.push(`trend=${validation.trend}`);
                        parts.push(`Validation#${validation.cycle}: ${valMetrics.join(' ')}`);
                    }
                    
                    el.textContent = parts.join(' | ');
                });
        }
        
        // Update submodels list
        function updateSubmodelsList() {
            fetch('/api/submodels')
                .then(response => response.json())
                .then(data => {
                    submodels = data;
                    const submodelsList = document.getElementById('submodels-list');
                    submodelsList.innerHTML = '';
                    
                    data.forEach(submodel => {
                        const submodelDiv = document.createElement('div');
                        submodelDiv.className = 'submodel';
                        submodelDiv.style.borderLeftColor = submodel.color;
                        
                        const nameDiv = document.createElement('div');
                        nameDiv.className = 'submodel-name';
                        nameDiv.textContent = submodel.name;
                        
                        const roleDiv = document.createElement('div');
                        roleDiv.className = 'submodel-role';
                        roleDiv.textContent = submodel.role;
                        
                        const capabilitiesDiv = document.createElement('div');
                        capabilitiesDiv.className = 'submodel-capabilities';
                        capabilitiesDiv.textContent = submodel.capabilities.join(' → ');
                        
                        submodelDiv.appendChild(nameDiv);
                        submodelDiv.appendChild(roleDiv);
                        submodelDiv.appendChild(capabilitiesDiv);
                        
                        submodelsList.appendChild(submodelDiv);
                    });
                });
        }

        function updateExternalList() {
            fetch('/api/external_agents')
                .then(response => response.json())
                .then(data => {
                    externals = data;
                    const externalList = document.getElementById('external-list');
                    externalList.innerHTML = '';

                    if (!data.length) {
                        externalList.textContent = 'None available';
                        return;
                    }

                    data.forEach(agent => {
                        const div = document.createElement('div');
                        div.className = 'submodel';
                        div.style.borderLeftColor = agent.color || '#888888';

                        const nameDiv = document.createElement('div');
                        nameDiv.className = 'submodel-name';
                        nameDiv.textContent = agent.name;

                        const roleDiv = document.createElement('div');
                        roleDiv.className = 'submodel-role';
                        roleDiv.textContent = agent.provider;

                        div.appendChild(nameDiv);
                        div.appendChild(roleDiv);
                        externalList.appendChild(div);
                    });
                });
        }
        
        // Update status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status;
                });
        }
        
        // Load messages
        function loadMessages() {
            fetch('/api/messages')
                .then(response => response.json())
                .then(messages => {
                    const messagesDiv = document.getElementById('messages');
                    messagesDiv.innerHTML = '';

                    messages.sort((a, b) => (a.id || 0) - (b.id || 0));
                    messages.forEach(msg => addMessage(msg));

                    lastMessageCount = messages.length;
                    if (messages.length) {
                        lastSeq = messages[messages.length - 1].id || 0;
                    }
                });
        }
        
        // Add message to chat
        function addMessage(message) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.style.borderLeftColor = message.color || '#3498db';

            if (message.agent_id === 'user') {
                messageDiv.classList.add('from-user');
            } else {
                messageDiv.classList.add('from-agent');
            }

            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';

            const senderSpan = document.createElement('span');
            senderSpan.className = 'sender';
            senderSpan.textContent = message.sender;

            const timeSpan = document.createElement('span');
            timeSpan.className = 'time';
            timeSpan.textContent = message.timestamp;

            metaDiv.appendChild(senderSpan);
            metaDiv.appendChild(timeSpan);

            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            contentDiv.textContent = message.message;

            messageDiv.appendChild(metaDiv);
            messageDiv.appendChild(contentDiv);
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Send message
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (message) {
                fetch('/api/message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        input.value = '';
                    }
                });
            }
        }
        
        // Handle Enter key
        document.getElementById('message-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize on load
        window.addEventListener('load', init);
    </script>
</body>
</html>
        ''')
        
        @self.app.after_request
        def after_request(response):
            """Add CORS headers for browser integration"""
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint for monitoring"""
            try:
                return jsonify({
                    'status': 'healthy',
                    'version': '2.0',
                    'components': {
                        'hub': True,
                        'neural_core': self._neural_core is not None,
                        'morphogenesis': self.morphogenesis_enabled,
                        'ollama': self.ollama_available,
                        'training': self.training_enabled
                    },
                    'stats': {
                        'processed': self.training_stats.get('processed', 0),
                        'concepts': len(self.concept_registry)
                    },
                    'timestamp': time.time()
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/submodels')
        def get_submodels():
            submodels = []
            for submodel_id, submodel in self.sam_submodels.items():
                submodels.append({
                    'id': submodel_id,
                    'name': submodel['name'],
                    'identifier': submodel.get('identifier'),
                    'specialty': submodel['specialty'],
                    'color': submodel['color'],
                    'capabilities': submodel['capabilities'],
                    'role': submodel['role']
                })
            return jsonify(submodels)

        @self.app.route('/api/external_agents')
        def get_external_agents():
            agents = []
            for _, agent in self.external_agents.items():
                agents.append({
                    'id': agent['id'],
                    'name': agent['name'],
                    'provider': agent['provider'],
                    'color': agent.get('color', '#888888')
                })
            return jsonify(agents)
        
        @self.app.route('/api/status')
        def get_status():
            try:
                deployed = "None"
                if len(self.conversation_history) > 0:
                    last_msg = list(self.conversation_history)[-1]
                    if last_msg['sender'] == 'You':
                        deployed = self.deploy_submodel(last_msg['message'])
                        deployed = self.sam_submodels[deployed]['name']

                external = "None"
                if self.last_external_used:
                    names = []
                    for ext_id in self.last_external_used:
                        if ext_id in self.external_agents:
                            names.append(self.external_agents[ext_id]['name'])
                    if names:
                        external = ", ".join(names)

                teach = "ON" if self.teach_sam_enabled else "OFF"
                distilled = "None"
                if self.last_distilled_ts:
                    distilled = datetime.fromtimestamp(self.last_distilled_ts).strftime('%H:%M:%S')
                
                return jsonify({
                    'status': (
                        f"👑 SAM-Head: Deployed {deployed} | "
                        f"🧩 Other Agents Replied: {external} | "
                        f"📚 Teach-SAM: {teach} (last {distilled}) | "
                        f"💬 {len(self.conversation_history)} messages"
                    ),
                    'typing_agent': self.typing_agent,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                return jsonify({'error': str(e), 'status': 'Error getting status'}), 500
        
        @self.app.route('/api/messages')
        def get_messages():
            try:
                messages = []
                for msg in list(self.conversation_history):
                    messages.append({
                        'id': msg.get('id'),
                        'sender': msg['sender'],
                        'message': msg['message'],
                        'timestamp': msg['timestamp'],
                        'ts': msg.get('ts'),
                        'agent_id': msg['agent_id'],
                        'color': msg['color']
                    })
                return jsonify(messages)
            except Exception as e:
                return jsonify({'error': str(e), 'messages': []}), 500

        @self.app.route('/api/agent_state')
        def get_agent_state():
            try:
                payload = {}
                now = time.time()
                for agent_id, st in self.agent_state.items():
                    merged = dict(st)
                    health = self.external_health.get(agent_id)
                    if health:
                        merged['disabled_until'] = health.get('disabled_until', 0.0)
                        remaining = max(0.0, float(health.get('disabled_until', 0.0)) - now)
                        merged['cooldown_remaining_s'] = round(remaining, 1)
                        merged['consecutive_timeouts'] = int(health.get('consecutive_timeouts', 0))
                    payload[agent_id] = merged

                payload['training'] = {
                    'queued': self.training_stats.get('queued', 0),
                    'processed': self.training_stats.get('processed', 0),
                    'last_status': self.training_stats.get('last_status'),
                    'last_ts': self.training_stats.get('last_ts'),
                    'muze_model_path': self.muze_model_path,
                    'muze_cli_path': self.muze_cli_path,
                    'muze_last': self.muze_last,
                    'cloud_backup_ts': self.last_cloud_backup_ts,
                }
                
                # Add validation metrics
                validation = self.get_validation_metrics()
                payload['validation'] = validation
                
                return jsonify(payload)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/message', methods=['POST'])
        def handle_message():
            try:
                data = request.json
                if not data:
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
                
                message = data.get('message', '')
                
                if message:
                    msg = message.strip()
                    if msg.startswith('/'):
                        self._handle_command(msg)
                    else:
                        self.add_message("You", msg, 'user')
                
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

    def _handle_command(self, cmdline):
        parts = cmdline.strip().split()
        cmd = parts[0].lower()

        if cmd == '/help':
            self.add_message('System', 'Commands: /help /clear /agents /teach on|off /mute <id> /unmute <id> /start /stop', 'system')
            return

        if cmd == '/clear':
            self.conversation_history.clear()
            self.add_message('System', 'Cleared messages.', 'system')
            return

        if cmd == '/agents':
            sam = ', '.join([a['name'] for _, a in self.sam_submodels.items()])
            ext = ', '.join([a['name'] for _, a in self.external_agents.items()]) or 'None'
            muted = ', '.join(sorted(self.muted_agents)) or 'None'
            self.add_message('System', f"SAM: {sam}\nOther: {ext}\nMuted IDs: {muted}", 'system')
            return

        if cmd == '/teach' and len(parts) >= 2:
            val = parts[1].lower()
            self.teach_sam_enabled = val in ('on', 'true', '1', 'yes')
            self.add_message('System', f"Teach-SAM set to {self.teach_sam_enabled}", 'system')
            return

        if cmd == '/mute' and len(parts) >= 2:
            agent_id = parts[1]
            self.muted_agents.add(agent_id)
            self.add_message('System', f"Muted {agent_id}", 'system')
            return

        if cmd == '/unmute' and len(parts) >= 2:
            agent_id = parts[1]
            self.muted_agents.discard(agent_id)
            self.add_message('System', f"Unmuted {agent_id}", 'system')
            return

        if cmd == '/start':
            self.auto_conversation_active = True
            self.add_message('System', 'Auto conversation started (SAM submodels).', 'system')
            return

        if cmd == '/stop':
            self.auto_conversation_active = False
            self.add_message('System', 'Auto conversation stopped.', 'system')
            return

        self.add_message('System', f"Unknown command: {cmdline}", 'system')

    # =================================================================
    # LATENT-SPACE MORPHOGENESIS (Concept Birth)
    # Dynamic creation of new latent dimensions when needed
    # =================================================================
    
    def check_morphogenesis_trigger(self, current_error):
        """
        Check if we should birth a new concept based on:
        1. Error above threshold
        2. Error not decreasing (stuck optimization)
        3. Not at max capacity
        
        Uses C implementation via neural core if available, otherwise Python fallback.
        """
        if not self.morphogenesis_enabled:
            return False
        
        # Try C implementation first
        if self._neural_core:
            try:
                self._neural_core.record_error(current_error)
                trigger = self._neural_core.check_morphogenesis_trigger(current_error)
                if trigger:
                    trend = self._neural_core.get_error_trend()
                    cost = self._neural_core.get_structure_cost()
                    print(f"🚨 C-morphogenesis trigger: error={current_error:.4f}, trend={trend:.4f}, cost={cost:.4f}")
                return trigger
            except Exception as e:
                print(f"⚠️ C-morphogenesis failed: {e}, using Python fallback")
        
        # Python fallback implementation
        # Record error
        self.error_history.append(current_error)
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Need minimum history
        if len(self.error_history) < 20:
            return False
        
        # Condition 1: Error above threshold
        if current_error < self.concept_birth_threshold:
            return False
        
        # Condition 2: Error not decreasing
        recent_errors = self.error_history[-20:]
        trend = self._compute_error_trend(recent_errors)
        if trend < -0.01:  # Error is decreasing
            return False
        
        # Condition 3: Check rank deficiency (simplified)
        rank_deficiency = self._estimate_rank_deficiency()
        if rank_deficiency <= 0:
            return False
        
        print(f"🚨 Python-morphogenesis trigger: error={current_error:.4f}, trend={trend:.4f}, deficiency={rank_deficiency}")
        return True
    
    def _compute_error_trend(self, errors):
        """Compute linear trend in errors"""
        n = len(errors)
        if n < 2:
            return 0.0
        
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(errors) / n
        
        numerator = sum((x[i] - mean_x) * (errors[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def _estimate_rank_deficiency(self):
        """
        Estimate rank deficiency in latent space
        Simplified version - full implementation uses SVD of curvature matrix
        """
        # Count active concepts
        active_concepts = len([c for c in self.concept_registry.values() if c.get('is_active', True)])
        
        # Estimate maximum useful dimensions based on training samples
        max_useful_dim = min(256, self.training_stats.get('processed', 0) // 10)
        
        if active_concepts >= max_useful_dim:
            return 0  # At capacity
        
        # Check for structural patterns in recent errors
        if len(self.error_history) >= 20:
            recent_variance = sum((e - sum(self.error_history[-20:])/20)**2 for e in self.error_history[-20:]) / 20
            if recent_variance > 0.01:  # High variance suggests missing structure
                return max(1, (max_useful_dim - active_concepts) // 4)
        
        return max(1, max_useful_dim - active_concepts) if active_concepts < max_useful_dim // 2 else 0
    
    def birth_concept(self, concept_name=None, context_trigger=None):
        """
        Birth a new concept (latent dimension)
        This expands the network's representational capacity
        
        Uses C implementation via neural core if available.
        """
        concept_id = f"concept_{len(self.concept_registry)}_{int(time.time())}"
        
        if not concept_name:
            concept_name = f"unnamed_concept_{len(self.concept_registry)}"
        
        # Try C implementation first
        if self._neural_core:
            try:
                success = self._neural_core.birth_concept(concept_name)
                if success:
                    # Also expand any active networks
                    if hasattr(self, '_network_manager') and self._network_manager:
                        for net_id in self._network_manager.list_networks():
                            current_dim = self._network_manager.get_network_info(net_id)
                            if current_dim and 'hidden_dims' in current_dim:
                                new_dim = current_dim['hidden_dims'][-1] + 1
                                self._network_manager.expand_latent_dim(net_id, new_dim)
                    
                    print(f"🌱 C-concept born: '{concept_name}' (ID: {concept_id})")
            except Exception as e:
                print(f"⚠️ C-concept birth failed: {e}")
        
        # Always create Python concept record
        concept = {
            'id': concept_id,
            'name': concept_name,
            'birth_time': time.time(),
            'birth_error': self.error_history[-1] if self.error_history else 0.0,
            'age': 0,
            'utility': 0.0,
            'is_active': True,
            'context_trigger': context_trigger,
            'specialization': self.current_context if hasattr(self, 'current_context') else 'general'
        }
        
        self.concept_registry[concept_id] = concept
        
        print(f"🌱 Concept born: '{concept_name}' (ID: {concept_id})")
        print(f"   Total concepts: {len(self.concept_registry)}")
        
        # Save concept to knowledge base
        self.save_to_knowledge_base({
            'kind': 'concept_birth',
            'concept_id': concept_id,
            'concept_name': concept_name,
            'birth_error': concept['birth_error'],
            'context': context_trigger,
            'timestamp': time.time()
        })
        
        return concept_id
    
    def prune_concept(self, concept_id):
        """
        Prune a concept that is no longer useful
        Implements concept death for structural regularization
        """
        if concept_id not in self.concept_registry:
            return False
        
        concept = self.concept_registry[concept_id]
        
        # Check minimum lifetime
        if concept['age'] < 10:
            print(f"⚠️ Cannot prune concept '{concept['name']}': too young (age={concept['age']})")
            return False
        
        # Check utility
        if concept['utility'] > 0.01:
            print(f"⚠️ Cannot prune concept '{concept['name']}': still useful (utility={concept['utility']:.4f})")
            return False
        
        concept['is_active'] = False
        concept['death_time'] = time.time()
        
        print(f"💀 Concept pruned: '{concept['name']}' (lived {concept['age']} updates)")
        
        return True
    
    def update_concept_utilities(self, current_error_reduction):
        """
        Update concept utilities based on how much they reduce error
        Called after each training step
        """
        for concept in self.concept_registry.values():
            if not concept['is_active']:
                continue
            
            concept['age'] += 1
            
            # Update utility based on error reduction since birth
            if concept['birth_error'] > 0:
                utility = max(0, (concept['birth_error'] - current_error_reduction) / concept['birth_error'])
                concept['utility'] = 0.9 * concept['utility'] + 0.1 * utility  # Smooth
    
    def get_morphogenesis_summary(self):
        """Get summary of concept registry"""
        active = [c for c in self.concept_registry.values() if c['is_active']]
        dead = [c for c in self.concept_registry.values() if not c['is_active']]
        
        return {
            'total_concepts': len(self.concept_registry),
            'active_concepts': len(active),
            'dead_concepts': len(dead),
            'avg_utility': sum(c['utility'] for c in active) / len(active) if active else 0,
            'concepts': [
                {
                    'name': c['name'],
                    'age': c['age'],
                    'utility': c['utility'],
                    'active': c['is_active']
                }
                for c in list(self.concept_registry.values())[-10:]  # Last 10
            ]
        }
    
    def run(self, host='127.0.0.1', port=8080, debug=False):
        """Run the Correct SAM Hub"""
        print(f"\n🚀 Starting Correct SAM Hub")
        print(f"🌐 URL: http://{host}:{port}")
        print("👑 SAM Head Model + Submodels")
        print("🤖 Each submodel does: Research → Self-RAG → Augmentation")
        print("💬 Correct SAM architecture")
        print("🛑 Ctrl+C to stop")
        print("=" * 50)
        
        # Start conversation system
        self.start_conversation_system()
        
        try:
            self.app.run(host=host, port=port, debug=False)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.getenv('SAM_HUB_PORT', '8080')))
    parser.add_argument('--no-teach', action='store_true')
    args = parser.parse_args()

    hub = CorrectSAMHub()
    if args.no_teach:
        hub.teach_sam_enabled = False
    hub.run(host=args.host, port=args.port)
