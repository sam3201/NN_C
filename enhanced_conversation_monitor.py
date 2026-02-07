#!/usr/bin/env python3
"""
Enhanced Conversation Monitor with Repetition Detection
Monitors repetition, coherence, and provides comprehensive analysis
"""

import re
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import huggingface LLM for enhanced analysis
try:
    from local_llm import local_llm, analyze_message_coherence as llm_analyze_coherence, get_improvement_suggestions
    LOCAL_LLM_AVAILABLE = local_llm.is_available()
    if LOCAL_LLM_AVAILABLE:
        print("✅ HuggingFace LLM loaded for conversation monitoring")
    else:
        print("⚠️ HuggingFace LLM available but not fully loaded")
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    print("⚠️ HuggingFace LLM not available, using basic monitoring")

@dataclass
class ConversationMetrics:
    """Comprehensive conversation metrics"""
    coherence_score: float
    repetition_score: float      # 0.0 to 1.0 (higher = less repetition)
    novelty_score: float        # 0.0 to 1.0 (higher = more novel)
    engagement_score: float     # 0.0 to 1.0
    overall_quality: float      # Weighted combination
    issues: List[str]
    suggestions: List[str]
    repetition_patterns: List[str]
    novelty_indicators: List[str]

class EnhancedConversationMonitor:
    """Enhanced monitor with repetition detection and comprehensive analysis"""
    
    def __init__(self, history_window=50):
        self.history_window = history_window
        self.message_history = deque(maxlen=history_window)
        self.word_frequency = defaultdict(int)
        self.phrase_frequency = defaultdict(int)
        self.topic_history = deque(maxlen=20)
        self.repetition_threshold = 0.3
        self.novelty_threshold = 0.5
        
        # Repetition patterns
        self.repetition_patterns = [
            r'(.{10,})\1+',  # Repeated phrases
            r'\b(\w+)\s+\1\b',  # Repeated words
            r'(.{5,})\s+\1+',  # Repeated short phrases
        ]
        
        # Incoherence patterns
        self.incoherence_patterns = [
            r'but.*actually',  # Contradictory statements
            r'and.*but.*and',  # Confusing conjunctions
            r'because.*but',   # Conflicting explanations
            r'however.*therefore',  # Logical contradictions
        ]
        
        # Novelty indicators
        self.novelty_keywords = [
            'new', 'interesting', 'different', 'unique', 'innovative',
            'surprising', 'unexpected', 'fresh', 'original', 'creative'
        ]
        
        # Engagement indicators
        self.engagement_positive = [
            'yes', 'agree', 'understand', 'interesting', 'good point',
            'tell me more', 'fascinating', 'excellent', 'great'
        ]
        
        self.engagement_negative = [
            'boring', 'confusing', 'unclear', 'repetitive', 'same thing',
            'already said', 'not following', 'lost'
        ]
    
    def analyze_message(self, message: str, context: Optional[List[str]] = None) -> ConversationMetrics:
        """Comprehensive message analysis"""
        issues = []
        suggestions = []
        
        # Basic coherence analysis
        coherence_score, coherence_issues = self._analyze_coherence_advanced(message, context)
        
        # Repetition analysis
        repetition_score, repetition_patterns = self._analyze_repetition(message)
        
        # Novelty analysis
        novelty_score, novelty_indicators = self._analyze_novelty(message, context)
        
        # Engagement analysis
        engagement_score = self._analyze_engagement(message, context)
        
        # Generate issues and suggestions
        issues.extend(self._generate_issues(repetition_score, novelty_score, engagement_score))
        suggestions.extend(self._generate_suggestions(issues))
        
        # Calculate overall quality
        overall_quality = (
            coherence_score * 0.3 +
            repetition_score * 0.25 +
            novelty_score * 0.25 +
            engagement_score * 0.2
        )
        
        # Update history
        self._update_history(message, overall_quality)
        
        return ConversationMetrics(
            coherence_score=coherence_score,
            repetition_score=repetition_score,
            novelty_score=novelty_score,
            engagement_score=engagement_score,
            overall_quality=overall_quality,
            issues=issues,
            suggestions=suggestions,
            repetition_patterns=repetition_patterns,
            novelty_indicators=novelty_indicators
        )
    
    def _analyze_coherence(self, message: str, context: Optional[List[str]]) -> float:
        """Analyze message coherence"""
        # Basic coherence checks
        coherence_score = 1.0
        
        # Grammar and structure
        if not message[0].isupper() if message else False:
            coherence_score -= 0.1
        
        if not message.endswith(('.','!','?')) if message else False:
            coherence_score -= 0.1
        
        # Context relevance
        if context:
            message_words = set(message.lower().split())
            context_words = set()
            for ctx in context[-3:]:
                context_words.update(ctx.lower().split())
            
            overlap = len(message_words & context_words)
            if overlap == 0:
                coherence_score -= 0.3
        
        return max(0.0, coherence_score)
    
    def _analyze_coherence_advanced(self, message: str, context: Optional[List[str]]) -> Tuple[float, List[str]]:
        """Analyze logical coherence with LLM assistance"""
        issues = []
        score = 1.0

        # Use local LLM for advanced coherence analysis if available
        if LOCAL_LLM_AVAILABLE and custom_llm:
            try:
                llm_analysis = custom_llm.analyze_coherence(message, context)
                llm_score = llm_analysis.get('coherence_score', 0.5)
                llm_issues = llm_analysis.get('issues', [])
                
                # Blend LLM score with rule-based score
                score = (score + llm_score) / 2
                issues.extend(llm_issues)
                
            except Exception as e:
                print(f"⚠️ Custom LLM coherence analysis failed: {e}")
                # Fall back to rule-based analysis
        
        # Rule-based coherence analysis (always available)
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
        """Detect contradictions between message and context"""
        contradictions = []
        message_lower = message.lower()
        
        for ctx in context[-2:]:  # Check last 2 context messages
            ctx_lower = ctx.lower()
            
            # Simple contradiction patterns
            if ("yes" in ctx_lower and "no" in message_lower) or ("no" in ctx_lower and "yes" in message_lower):
                contradictions.append("Direct yes/no contradiction")
            
            if ("true" in ctx_lower and "false" in message_lower) or ("false" in ctx_lower and "true" in message_lower):
                contradictions.append("True/false contradiction")
            
            # Weather contradictions
            weather_opposites = [
                ("sunny", "rainy"), ("rainy", "sunny"),
                ("hot", "cold"), ("cold", "hot"),
                ("clear", "cloudy"), ("cloudy", "clear")
            ]
            
            for word1, word2 in weather_opposites:
                if word1 in ctx_lower and word2 in message_lower:
                    contradictions.append(f"Weather contradiction: {word1} vs {word2}")
        
        return contradictions
    
    def _analyze_repetition(self, message: str) -> Tuple[float, List[str]]:
        repetition_patterns = []
        repetition_score = 1.0
        
        # Check for repeated patterns
        for pattern in self.repetition_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                repetition_patterns.extend(matches)
                repetition_score -= 0.2 * len(matches)
        
        # Check against history
        message_lower = message.lower()
        historical_repetitions = 0
        
        for hist_msg in self.message_history:
            hist_lower = hist_msg.lower()
            
            # Exact match
            if message_lower == hist_lower:
                historical_repetitions += 1
                repetition_score -= 0.3
            # High similarity
            elif self._calculate_similarity(message_lower, hist_lower) > 0.8:
                historical_repetitions += 1
                repetition_score -= 0.1
        
        # Word frequency analysis
        words = message.lower().split()
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        # Penalize words used multiple times in same message
        for word, count in word_counts.items():
            if count > 2:
                repetition_score -= 0.1 * (count - 2)
                repetition_patterns.append(f"Repeated word: '{word}' ({count} times)")
        
        return max(0.0, repetition_score), repetition_patterns
    
    def _analyze_novelty(self, message: str, context: Optional[List[str]]) -> Tuple[float, List[str]]:
        """Analyze novelty and new information"""
        novelty_indicators = []
        novelty_score = 0.5  # Base score
        
        # Check for novelty keywords
        message_lower = message.lower()
        for keyword in self.novelty_keywords:
            if keyword in message_lower:
                novelty_indicators.append(f"Novelty indicator: '{keyword}'")
                novelty_score += 0.1
        
        # Check for new concepts (not in recent history)
        message_words = set(message.lower().split())
        historical_words = set()
        
        for hist_msg in self.message_history:
            historical_words.update(hist_msg.lower().split())
        
        new_words = message_words - historical_words
        if len(new_words) > 2:
            novelty_score += 0.2
            novelty_indicators.append(f"New concepts: {len(new_words)} new words")
        
        # Check for question patterns (indicates curiosity)
        if '?' in message and message.count('?') > 1:
            novelty_score += 0.1
            novelty_indicators.append("Multiple questions - high engagement")
        
        # Check for complex sentences
        sentences = re.split(r'[.!?]+', message)
        complex_sentences = [s for s in sentences if len(s.split()) > 15]
        if complex_sentences:
            novelty_score += 0.1
            novelty_indicators.append(f"Complex sentences: {len(complex_sentences)}")
        
        return min(1.0, novelty_score), novelty_indicators
    
    def _analyze_engagement(self, message: str, context: Optional[List[str]]) -> float:
        """Analyze engagement level"""
        engagement_score = 0.5  # Base score
        
        message_lower = message.lower()
        
        # Positive engagement indicators
        for indicator in self.engagement_positive:
            if indicator in message_lower:
                engagement_score += 0.1
        
        # Negative engagement indicators
        for indicator in self.engagement_negative:
            if indicator in message_lower:
                engagement_score -= 0.2
        
        # Message length (moderate length is more engaging)
        word_count = len(message.split())
        if 5 <= word_count <= 20:
            engagement_score += 0.1
        elif word_count > 30:
            engagement_score -= 0.1
        
        # Questions indicate engagement
        if '?' in message:
            engagement_score += 0.1
        
        return max(0.0, min(1.0, engagement_score))
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_issues(self, repetition_score: float, novelty_score: float, engagement_score: float) -> List[str]:
        """Generate issues based on scores"""
        issues = []
        
        if repetition_score < 0.5:
            issues.append("High repetition detected")
        
        if novelty_score < 0.3:
            issues.append("Low novelty - repetitive content")
        
        if engagement_score < 0.3:
            issues.append("Low engagement indicators")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        for issue in issues:
            if "repetition" in issue.lower():
                suggestions.append("Try to vary your language and avoid repeating phrases")
            if "novelty" in issue.lower():
                suggestions.append("Introduce new concepts or different perspectives")
            if "engagement" in issue.lower():
                suggestions.append("Use more engaging language and ask questions")
        
        return list(set(suggestions))
    
    def _update_history(self, message: str, quality_score: float):
        """Update conversation history"""
        self.message_history.append(message)
        
        # Update word frequency
        words = message.lower().split()
        for word in words:
            self.word_frequency[word] += 1
        
        # Update topic history (simplified)
        if quality_score > 0.7:
            self.topic_history.append(message[:50])  # First 50 chars as topic proxy
    
    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive monitoring report"""
        if not self.message_history:
            return {"error": "No conversation history"}
        
        # Calculate averages
        recent_messages = list(self.message_history)[-20:]
        
        # Word frequency analysis
        top_words = sorted(self.word_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Repetition analysis
        total_repetitions = sum(1 for i, msg in enumerate(recent_messages) 
                              if i > 0 and self._calculate_similarity(msg.lower(), recent_messages[i-1].lower()) > 0.7)
        
        return {
            'total_messages': len(self.message_history),
            'recent_messages': len(recent_messages),
            'repetition_rate': total_repetitions / len(recent_messages) if recent_messages else 0,
            'top_words': top_words,
            'topic_diversity': len(set(self.topic_history)) / len(self.topic_history) if self.topic_history else 0,
            'monitoring_window': self.history_window,
            'last_updated': datetime.now().isoformat()
        }

# Global enhanced monitor
enhanced_monitor = EnhancedConversationMonitor()

def analyze_conversation_enhanced(message: str, context: Optional[List[str]] = None) -> ConversationMetrics:
    """Analyze conversation with enhanced monitoring"""
    return enhanced_monitor.analyze_message(message, context)

def get_enhanced_monitoring_report() -> Dict:
    """Get comprehensive monitoring report"""
    return enhanced_monitor.get_comprehensive_report()
