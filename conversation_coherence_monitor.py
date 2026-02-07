#!/usr/bin/env python3
"""
Conversation Coherence Monitor for SAM 2.0
Monitors chat messages for coherence and provides loss/reward signals
"""

import re
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CoherenceMetrics:
    """Metrics for conversation coherence"""
    coherence_score: float  # 0.0 to 1.0
    grammar_score: float    # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    clarity_score: float    # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    overall_score: float   # Weighted combination
    issues: List[str]      # Specific issues found
    suggestions: List[str] # Improvement suggestions

class ConversationCoherenceMonitor:
    """Monitors conversation coherence and provides learning signals"""
    
    def __init__(self):
        self.coherence_history = []
        self.baseline_scores = {
            'grammar': 0.8,
            'relevance': 0.7,
            'clarity': 0.75,
            'completeness': 0.8
        }
        
        # Coherence patterns
        self.incoherence_patterns = [
            r'\b(suddenly|randomly|out of nowhere)\b.*\b(because|so|therefore)\b',
            r'\b(because|so|therefore)\b.*\b(unrelated|different|separate)\b',
            r'\b(I mean|actually|wait)\b.*\b(no|never)\b',
            r'\b(yes|no)\b.*\b(but actually|however)\b.*\b(no|yes)\b',
        ]
        
        # Grammar issues
        self.grammar_patterns = [
            r'\b(I|you|he|she|it|we|they)\s+\w+s\b',  # Subject-verb agreement
            r'\b(a|an)\s+([aeiouAEIOU])',  # Article issues
            r'\b(the|a|an)\s+\w+\s+\w+\s+(the|a|an)\b',  # Double articles
            r'\.{2,}',  # Multiple periods
            r'[!?]{3,}',  # Excessive punctuation
        ]
        
        # Incomplete sentence patterns
        self.incomplete_patterns = [
            r'\b(because|since|although|while|when|if)\s*$',
            r'\b(and|or|but)\s*$',
            r'^\s*\.\.\.\s*$',
            r'^\s*[A-Z][a-z]*\s*$'  # Single word
        ]
    
    def analyze_message(self, message: str, context: Optional[List[str]] = None) -> CoherenceMetrics:
        """Analyze a single message for coherence"""
        issues = []
        suggestions = []
        
        # Grammar score
        grammar_score, grammar_issues = self._analyze_grammar(message)
        issues.extend(grammar_issues)
        
        # Relevance score (if context provided)
        if context:
            relevance_score, relevance_issues = self._analyze_relevance(message, context)
            issues.extend(relevance_issues)
        else:
            relevance_score = 0.8  # Default for first message
        
        # Clarity score
        clarity_score, clarity_issues = self._analyze_clarity(message)
        issues.extend(clarity_issues)
        
        # Completeness score
        completeness_score, completeness_issues = self._analyze_completeness(message)
        issues.extend(completeness_issues)
        
        # Coherence score (logical flow)
        coherence_score, coherence_issues = self._analyze_coherence(message, context)
        issues.extend(coherence_issues)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(issues)
        
        # Calculate overall score
        overall_score = (
            grammar_score * 0.25 +
            relevance_score * 0.25 +
            clarity_score * 0.25 +
            completeness_score * 0.25
        )
        
        metrics = CoherenceMetrics(
            coherence_score=coherence_score,
            grammar_score=grammar_score,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )
        
        # Store in history
        self.coherence_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'metrics': metrics
        })
        
        return metrics
    
    def _analyze_grammar(self, message: str) -> Tuple[float, List[str]]:
        """Analyze grammar quality"""
        issues = []
        score = 1.0
        
        # Check grammar patterns
        for pattern in self.grammar_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                issues.append(f"Grammar issue detected: {pattern}")
                score -= 0.1 * len(matches)
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', message)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) < 3:
                issues.append(f"Very short sentence: '{sentence}'")
                score -= 0.05
        
        # Check capitalization
        if message and not message[0].isupper():
            issues.append("Message starts with lowercase")
            score -= 0.05
        
        return max(0.0, score), issues
    
    def _analyze_relevance(self, message: str, context: List[str]) -> Tuple[float, List[str]]:
        """Analyze relevance to conversation context"""
        issues = []
        score = 1.0
        
        if not context:
            return score, issues
        
        # Simple keyword overlap analysis
        message_words = set(message.lower().split())
        context_words = set()
        for ctx_msg in context[-3:]:  # Last 3 messages
            context_words.update(ctx_msg.lower().split())
        
        # Calculate overlap
        overlap = len(message_words & context_words)
        if overlap == 0:
            issues.append("No relevant keywords from context")
            score = 0.3
        elif overlap < 2:
            issues.append("Low relevance to conversation")
            score = 0.6
        else:
            score = min(1.0, 0.6 + (overlap * 0.1))
        
        return score, issues
    
    def _analyze_clarity(self, message: str) -> Tuple[float, List[str]]:
        """Analyze message clarity"""
        issues = []
        score = 1.0
        
        # Check for ambiguous terms
        ambiguous_terms = ['thing', 'stuff', 'something', 'anything', 'nothing']
        for term in ambiguous_terms:
            if term in message.lower():
                issues.append(f"Ambiguous term: '{term}'")
                score -= 0.1
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', message)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            issues.append(f"Long sentences detected: {len(long_sentences)}")
            score -= 0.1 * len(long_sentences)
        
        # Check for clarity indicators
        if 'I mean' in message.lower():
            issues.append("Self-correction indicator")
            score -= 0.1
        
        return max(0.0, score), issues
    
    def _analyze_completeness(self, message: str) -> Tuple[float, List[str]]:
        """Analyze message completeness"""
        issues = []
        score = 1.0
        
        # Check for incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues.append(f"Incomplete sentence pattern: {pattern}")
                score -= 0.2
        
        # Check for trailing punctuation
        if message and not message.endswith(('.', '!', '?')):
            issues.append("Missing ending punctuation")
            score -= 0.1
        
        # Check for very short messages
        if len(message.split()) < 3:
            issues.append("Very short message")
            score -= 0.2
        
        return max(0.0, score), issues
    
    def _analyze_coherence(self, message: str, context: Optional[List[str]]) -> Tuple[float, List[str]]:
        """Analyze logical coherence"""
        issues = []
        score = 1.0
        
        # Check incoherence patterns
        for pattern in self.incoherence_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                issues.append(f"Incoherent logic pattern: {pattern}")
                score -= 0.2
        
        # Check for contradictions
        if context:
            contradictions = self._detect_contradictions(message, context[-1:])
            if contradictions:
                issues.extend(contradictions)
                score -= 0.3
        
        return max(0.0, score), issues
    
    def _detect_contradictions(self, message: str, context: List[str]) -> List[str]:
        """Detect contradictions with context"""
        contradictions = []
        
        # Simple contradiction detection
        positive_words = ['yes', 'true', 'correct', 'agree', 'definitely']
        negative_words = ['no', 'false', 'incorrect', 'disagree', 'definitely not']
        
        message_lower = message.lower()
        
        for ctx_msg in context:
            ctx_lower = ctx_msg.lower()
            
            # Check for direct contradictions
            pos_in_msg = any(word in message_lower for word in positive_words)
            neg_in_msg = any(word in message_lower for word in negative_words)
            pos_in_ctx = any(word in ctx_lower for word in positive_words)
            neg_in_ctx = any(word in ctx_lower for word in negative_words)
            
            if (pos_in_msg and neg_in_ctx) or (neg_in_msg and pos_in_ctx):
                contradictions.append("Potential contradiction with context")
        
        return contradictions
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues"""
        suggestions = []
        
        for issue in issues:
            if 'Grammar issue' in issue:
                suggestions.append("Review grammar and sentence structure")
            elif 'short sentence' in issue:
                suggestions.append("Provide more detailed explanations")
            elif 'Ambiguous term' in issue:
                suggestions.append("Use more specific language")
            elif 'Long sentences' in issue:
                suggestions.append("Break down complex ideas into shorter sentences")
            elif 'Incomplete' in issue:
                suggestions.append("Complete your thoughts and use proper punctuation")
            elif 'contradiction' in issue:
                suggestions.append("Ensure consistency with previous messages")
            elif 'relevance' in issue:
                suggestions.append("Stay on topic and reference the conversation")
        
        return list(set(suggestions))  # Remove duplicates
    
    def calculate_loss_signal(self, metrics: CoherenceMetrics) -> float:
        """Calculate loss signal for training (lower is better)"""
        # Convert to loss: 1.0 - score
        base_loss = 1.0 - metrics.overall_score
        
        # Penalty for multiple issues
        issue_penalty = len(metrics.issues) * 0.05
        
        # Penalty for very low scores
        if metrics.overall_score < 0.5:
            base_loss += 0.2
        
        total_loss = min(1.0, base_loss + issue_penalty)
        return total_loss
    
    def calculate_reward_signal(self, metrics: CoherenceMetrics) -> float:
        """Calculate reward signal for training (higher is better)"""
        # Base reward from overall score
        base_reward = metrics.overall_score
        
        # Bonus for no issues
        if len(metrics.issues) == 0:
            base_reward += 0.1
        
        # Bonus for high coherence
        if metrics.coherence_score > 0.8:
            base_reward += 0.1
        
        # Bonus for improvement over baseline
        improvement = metrics.overall_score - 0.7  # Assuming 0.7 baseline
        if improvement > 0:
            base_reward += improvement * 0.5
        
        total_reward = min(1.0, base_reward)
        return total_reward
    
    def get_coherence_report(self) -> Dict:
        """Get comprehensive coherence report"""
        if not self.coherence_history:
            return {"error": "No conversation history"}
        
        recent_metrics = [entry['metrics'] for entry in self.coherence_history[-10:]]
        
        avg_scores = {
            'overall': np.mean([m.overall_score for m in recent_metrics]),
            'grammar': np.mean([m.grammar_score for m in recent_metrics]),
            'relevance': np.mean([m.relevance_score for m in recent_metrics]),
            'clarity': np.mean([m.clarity_score for m in recent_metrics]),
            'completeness': np.mean([m.completeness_score for m in recent_metrics]),
            'coherence': np.mean([m.coherence_score for m in recent_metrics])
        }
        
        common_issues = {}
        for entry in self.coherence_history[-20:]:
            for issue in entry['metrics'].issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        return {
            'average_scores': avg_scores,
            'total_messages': len(self.coherence_history),
            'common_issues': sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5],
            'trend': 'improving' if len(self.coherence_history) > 1 else 'stable'
        }

# Global coherence monitor instance
coherence_monitor = ConversationCoherenceMonitor()

def analyze_conversation_coherence(message: str, context: Optional[List[str]] = None) -> CoherenceMetrics:
    """Analyze conversation coherence and return metrics"""
    return coherence_monitor.analyze_message(message, context)

def get_coherence_loss_and_reward(message: str, context: Optional[List[str]] = None) -> Tuple[float, float]:
    """Get both loss and reward signals for a message"""
    metrics = analyze_conversation_coherence(message, context)
    loss = coherence_monitor.calculate_loss_signal(metrics)
    reward = coherence_monitor.calculate_reward_signal(metrics)
    return loss, reward

def get_coherence_report() -> Dict:
    """Get coherence monitoring report"""
    return coherence_monitor.get_coherence_report()
