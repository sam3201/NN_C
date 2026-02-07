#!/usr/bin/env python3
"""
Enhanced Web Search Integration for SAM System
Uses dedicated Google account for web searches and data persistence
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, urljoin
import time

class SAMWebSearch:
    """Enhanced web search using SAM's dedicated Google account"""

    def __init__(self, google_drive_integration=None):
        self.google_drive = google_drive_integration
        self.search_history = []
        self.max_history = 100

        # Get dedicated account credentials
        self.sam_email = os.getenv('GOOGLE_ACCOUNT', 'sam.ai.system.agi@gmail.com')
        self.sam_password = os.getenv('GOOGLE_PASSWORD', '')

        print("🔍 SAM Web Search initialized with dedicated account")

    def search(self, query: str, save_to_drive: bool = True) -> Dict:
        """Perform web search using SAM's dedicated infrastructure"""
        try:
            print(f"🔍 SAM Web Search: {query}")

            # Create search session
            search_session = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'account': self.sam_email,
                'results': [],
                'status': 'in_progress'
            }

            # Perform search using multiple sources
            results = self._multi_source_search(query)

            # Enhance results with SAM analysis
            enhanced_results = self._enhance_with_sam_analysis(query, results)

            search_session['results'] = enhanced_results
            search_session['status'] = 'completed'
            search_session['result_count'] = len(enhanced_results)

            # Save to Google Drive if available
            if save_to_drive and self.google_drive:
                self._save_search_to_drive(search_session)

            # Add to history
            self._add_to_history(search_session)

            return {
                'query': query,
                'results': enhanced_results,
                'source': 'sam_dedicated_search',
                'account': self.sam_email,
                'timestamp': search_session['timestamp'],
                'saved_to_drive': save_to_drive and self.google_drive is not None
            }

        except Exception as e:
            print(f"❌ SAM Web Search failed: {e}")
            return {
                'query': query,
                'error': str(e),
                'source': 'sam_dedicated_search',
                'timestamp': datetime.now().isoformat()
            }

    def _multi_source_search(self, query: str) -> List[Dict]:
        """Search using multiple sources for comprehensive results"""
        results = []

        try:
            # Primary: Use requests to search common sources
            # Note: In production, you'd integrate with proper search APIs

            # DuckDuckGo search (privacy-focused)
            ddg_results = self._search_duckduckgo(query)
            results.extend(ddg_results)

            # Additional sources could include:
            # - Google Custom Search API (requires API key)
            # - Bing Search API
            # - Local knowledge base search

        except Exception as e:
            print(f"⚠️ Multi-source search error: {e}")

        return results[:10]  # Limit to top 10 results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo (privacy-focused)"""
        try:
            # Use DuckDuckGo's instant answer API
            url = f"https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()

                results = []

                # Add instant answer if available
                if data.get('Answer'):
                    results.append({
                        'title': 'Instant Answer',
                        'content': data['Answer'],
                        'url': data.get('AnswerURL', ''),
                        'source': 'duckduckgo_instant',
                        'type': 'answer'
                    })

                # Add abstract if available
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'Abstract'),
                        'content': data['Abstract'],
                        'url': data.get('AbstractURL', ''),
                        'source': 'duckduckgo_abstract',
                        'type': 'summary'
                    })

                # Add related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', 'Related Topic'),
                            'content': topic['Text'][:200] + '...',
                            'url': topic.get('FirstURL', ''),
                            'source': 'duckduckgo_related',
                            'type': 'related'
                        })

                return results

        except Exception as e:
            print(f"⚠️ DuckDuckGo search error: {e}")

        return []

    def _enhance_with_sam_analysis(self, query: str, results: List[Dict]) -> List[Dict]:
        """Enhance search results with SAM analysis"""
        enhanced = []

        for result in results:
            # Add SAM metadata
            enhanced_result = result.copy()
            enhanced_result['sam_analysis'] = {
                'relevance_score': self._calculate_relevance(query, result),
                'credibility': self._assess_credibility(result),
                'analyzed_by': 'sam_web_search',
                'analysis_timestamp': datetime.now().isoformat()
            }
            enhanced.append(enhanced_result)

        return enhanced

    def _calculate_relevance(self, query: str, result: Dict) -> float:
        """Calculate relevance score for search result"""
        try:
            query_words = set(query.lower().split())
            content_words = set(result.get('content', '').lower().split())
            title_words = set(result.get('title', '').lower().split())

            # Simple word overlap scoring
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words))

            score = (title_overlap * 2 + content_overlap) / len(query_words)
            return min(score, 1.0)  # Cap at 1.0

        except:
            return 0.5  # Default medium relevance

    def _assess_credibility(self, result: Dict) -> str:
        """Assess credibility of search result"""
        url = result.get('url', '')

        if not url:
            return 'unknown'

        try:
            domain = urlparse(url).netloc.lower()

            # Known credible domains
            credible_domains = [
                'wikipedia.org', 'edu', 'gov', 'ac.uk', 'ac.jp',
                'nature.com', 'science.org', 'arxiv.org'
            ]

            if any(cred_domain in domain for cred_domain in credible_domains):
                return 'high'

            # Questionable sources
            questionable = ['fake', 'conspiracy', 'hoax']
            if any(q in domain for q in questionable):
                return 'low'

            return 'medium'

        except:
            return 'unknown'

    def _save_search_to_drive(self, search_session: Dict):
        """Save search results to Google Drive"""
        try:
            if not self.google_drive:
                return

            # Create search results file
            filename = f"search_{int(time.time())}.json"

            # Write search data to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                # Sanitize for privacy - remove any sensitive data
                sanitized_session = search_session.copy()
                sanitized_session['account'] = 'sam.ai.system@gmail.com (sanitized)'

                json.dump(sanitized_session, f, indent=2)
                temp_file = f.name

            # Upload to Google Drive
            drive_filename = f"sam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.google_drive.upload_file(temp_file, drive_filename)

            # Clean up temp file
            os.unlink(temp_file)

            print(f"💾 Search results saved to Google Drive: {drive_filename}")

        except Exception as e:
            print(f"⚠️ Failed to save search to Drive: {e}")

    def _add_to_history(self, search_session: Dict):
        """Add search to history"""
        self.search_history.append(search_session)
        if len(self.search_history) > self.max_history:
            self.search_history.pop(0)

    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get recent search history"""
        return self.search_history[-limit:]

    def clear_history(self):
        """Clear search history"""
        self.search_history.clear()
        print("🧹 Search history cleared")

# Global instance for SAM system integration
sam_web_search = None

def initialize_sam_web_search(google_drive=None):
    """Initialize SAM web search with Google Drive integration"""
    global sam_web_search
    sam_web_search = SAMWebSearch(google_drive)
    return sam_web_search

def search_web_with_sam(query: str, save_to_drive: bool = True) -> Dict:
    """Perform web search using SAM's dedicated infrastructure"""
    global sam_web_search
    if not sam_web_search:
        sam_web_search = SAMWebSearch()

    return sam_web_search.search(query, save_to_drive)
