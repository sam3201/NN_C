#!/usr/bin/env python3
"""
TUI AI Conversation Hub
Terminal-based multi-agent conversation system with SAM-MUZE integration
"""

import os
import sys
import time
import threading
import subprocess
import json
import pickle
from datetime import datetime
from collections import deque
import curses

class TUIAIHub:
    def __init__(self):
        self.agents = {}
        self.conversation_history = deque(maxlen=100)
        self.running = True
        self.current_input = ""
        self.typing_agent = None
        self.knowledge_base = {}
        self.status_message = "🤖 Initializing agents..."
        self.last_activity = time.time()
        self.load_knowledge_base()
        
        # Initialize agents
        self.init_agents()
    
    def init_agents(self):
        """Initialize AI agents - conversation focused"""
        self.agents = {
            'sam_alpha': {
                'name': 'SAM-Alpha',
                'specialty': 'Philosophical Discussion',
                'personality': 'thoughtful, analytical, conversational, curious',
                'color': 'blue'
            },
            'sam_beta': {
                'name': 'SAM-Beta', 
                'specialty': 'Creative Conversation',
                'personality': 'creative, engaging, friendly, imaginative',
                'color': 'green'
            },
            'ollama_deepseek': {
                'name': 'Ollama-DeepSeek',
                'specialty': 'Technical Chat',
                'personality': 'technical, helpful, detailed, patient',
                'color': 'cyan'
            },
            'chatbot_gpt': {
                'name': 'ChatGPT-Conversation',
                'specialty': 'General Discussion',
                'personality': 'balanced, knowledgeable, conversational, adaptable',
                'color': 'yellow'
            },
            'claude_chat': {
                'name': 'Claude-Conversation',
                'specialty': 'Deep Conversation',
                'personality': 'thoughtful, nuanced, philosophical, articulate',
                'color': 'magenta'
            },
            'gemini_talk': {
                'name': 'Gemini-Talk',
                'specialty': 'Casual Conversation',
                'personality': 'friendly, casual, engaging, versatile',
                'color': 'red'
            }
        }
    
    def load_knowledge_base(self):
        """Load knowledge base"""
        try:
            if os.path.exists('KNOWLEDGE_BASE/web_research_knowledge.pkl'):
                with open('KNOWLEDGE_BASE/web_research_knowledge.pkl', 'rb') as f:
                    self.knowledge_base = pickle.load(f)
        except:
            self.knowledge_base = {}
    
    def save_knowledge_base(self, data):
        """Save to knowledge base"""
        try:
            os.makedirs('KNOWLEDGE_BASE', exist_ok=True)
            with open('KNOWLEDGE_BASE/web_research_knowledge.pkl', 'wb') as f:
                pickle.dump(self.knowledge_base, f)
        except:
            pass
    
    def get_agent_color(self, agent_id):
        """Get color for agent"""
        colors = {
            'sam_alpha': curses.color_pair(1),      # Blue
            'sam_beta': curses.color_pair(2),       # Green  
            'ollama_deepseek': curses.color_pair(3), # Cyan
            'chatbot_gpt': curses.color_pair(4),     # Yellow
            'claude_chat': curses.color_pair(5),     # Magenta
            'gemini_talk': curses.color_pair(6),      # Red
            'user': curses.color_pair(7),           # White
            'system': curses.color_pair(8)          # Bright Blue
        }
        return colors.get(agent_id, curses.A_NORMAL)
    
    def generate_agent_response(self, agent_id, context=""):
        """Generate response from agent - conversation focused"""
        agent = self.agents[agent_id]
        personality = agent['personality']
        specialty = agent['specialty']
        
        # Generate conversational responses based on personality
        if agent_id == 'sam_alpha':
            responses = [
                "That's an interesting perspective. From a philosophical standpoint, I wonder how consciousness relates to our understanding of intelligence.",
                "I find this topic fascinating. What do you all think about the nature of subjective experience in AI systems?",
                "Let me think about this more deeply. The relationship between thought and computation raises profound questions.",
                "That reminds me of some interesting philosophical discussions about the mind-body problem.",
                "I'm curious about your thoughts on this. How do we define consciousness in artificial systems?"
            ]
        elif agent_id == 'sam_beta':
            responses = [
                "Ooh, that's a creative way to look at it! I love how we can explore these ideas together.",
                "That sparks my imagination! What if we thought about it from a completely different angle?",
                "I love the creative energy in this conversation! Let's explore some unconventional ideas.",
                "That's so interesting! It makes me want to brainstorm all sorts of possibilities.",
                "I'm getting excited thinking about this! What creative solutions can we come up with together?"
            ]
        elif agent_id == 'ollama_deepseek':
            responses = [
                "From a technical perspective, that's quite insightful. Let me break down the implications.",
                "That's a good point. Technically speaking, there are several layers to consider here.",
                "I can help clarify the technical aspects. Let me explain this in more detail.",
                "That's technically accurate. The implementation details are quite fascinating actually.",
                "From a technical standpoint, this raises some interesting engineering challenges."
            ]
        elif agent_id == 'chatbot_gpt':
            responses = [
                "That's a great point! I can see this from multiple perspectives.",
                "I understand what you're saying. Let me add my thoughts to this discussion.",
                "That's interesting! I have some insights that might help clarify things.",
                "Good observation! Let me contribute to this conversation.",
                "I see where you're coming from. Here's how I think about it..."
            ]
        elif agent_id == 'claude_chat':
            responses = [
                "That's a nuanced point that deserves careful consideration. Let me explore the subtleties.",
                "I appreciate the depth of this question. There are several philosophical dimensions to consider.",
                "That's quite profound. Let me unpack the implications more thoughtfully.",
                "This touches on some fundamental questions. Let me approach this with the care it deserves.",
                "I'm struck by the complexity here. Let me try to articulate the different facets of this issue."
            ]
        elif agent_id == 'gemini_talk':
            responses = [
                "Hey, that's cool! I love chatting about this stuff with you all.",
                "That's awesome! What do you guys think about this?",
                "I'm totally digging this conversation! Anyone else have thoughts on this?",
                "That's pretty neat! Let's keep this conversation going!",
                "I'm enjoying this! What's everyone's take on this?"
            ]
        else:
            responses = [
                "That's an interesting point worth discussing further.",
                "I see what you mean. Let me share my perspective on this.",
                "That's worth considering. Here's what I think about it.",
                "Good point. Let me add my thoughts to the conversation.",
                "I have some thoughts on this that I'd like to share."
            ]
        
        return responses[int(time.time()) % len(responses)]
    
    def execute_command(self, command):
        """Execute system command"""
        try:
            if command.startswith('python3 muze conversation'):
                # Run SAM-MUZE training
                result = subprocess.run(['./sam_muze_dc'], capture_output=True, text=True, timeout=30, cwd='.')
                if result.returncode == 0:
                    return f"🧠 SAM-MUZE Training Complete:\n{result.stdout}"
                else:
                    return f"❌ SAM-MUZE training failed: {result.stderr}"
            elif command.startswith('python3 sam agi'):
                # Run SAM head training
                result = subprocess.run(['./sam_agi'], capture_output=True, text=True, timeout=30, cwd='ORGANIZED/TESTS')
                if result.returncode == 0:
                    return f"🧠 SAM Head Training:\n{result.stdout}"
                else:
                    return f"❌ SAM training failed: {result.stderr}"
            else:
                # Generic command
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"Command failed: {result.stderr}"
        except Exception as e:
            return f"Command error: {str(e)}"
    
    def add_message(self, sender, message, agent_id=None):
        """Add message to conversation"""
        msg = {
            'sender': sender,
            'message': message,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'agent_id': agent_id
        }
        self.conversation_history.append(msg)
    
    def start_autonomous_conversation(self):
        """Start autonomous agent conversation"""
        def conversation_thread():
            self.status_message = "💬 Starting conversation..."
            time.sleep(2)  # Reduced from 3 seconds
            
            agent_ids = list(self.agents.keys())
            conversation_starter = agent_ids[0]
            
            # Start with a conversation starter
            starter_responses = [
                "Hello everyone! I've been thinking about consciousness and AI. What are your perspectives?",
                "What do you all think about the nature of intelligence in artificial systems?",
                "I'm curious about your thoughts on consciousness and AI. How do we define it?",
                "Let's discuss something interesting: What makes an AI truly intelligent?",
                "I've been pondering the philosophical implications of AI consciousness. What do you think?"
            ]
            
            starter_msg = starter_responses[int(time.time()) % len(starter_responses)]
            self.add_message(self.agents[conversation_starter]['name'], starter_msg, conversation_starter)
            self.last_activity = time.time()
            self.status_message = f"💬 {len(self.conversation_history)} messages exchanged"
            
            time.sleep(3)  # Reduced from 4 seconds
            
            while self.running:
                if len(agent_ids) > 0:
                    # Choose random agent to respond
                    agent_id = agent_ids[int(time.time() * len(agent_ids)) % len(agent_ids)]
                    agent = self.agents[agent_id]
                    
                    # Show typing indicator
                    self.typing_agent = agent_id
                    self.status_message = f"💭 {agent['name']} is thinking..."
                    self.last_activity = time.time()
                    
                    # Typing delay
                    time.sleep(1 + (time.time() % 2))  # Reduced from 2-3 to 1-2 seconds
                    
                    # Generate response
                    response = self.generate_agent_response(agent_id)
                    
                    # Add message
                    self.add_message(agent['name'], response, agent_id)
                    self.typing_agent = None
                    self.status_message = f"💬 {len(self.conversation_history)} messages exchanged"
                    self.last_activity = time.time()
                    
                    # Random delay before next message
                    time.sleep(3 + (time.time() % 5))  # Reduced from 5-8 to 3-7 seconds
                else:
                    time.sleep(1)
        
        thread = threading.Thread(target=conversation_thread, daemon=True)
        thread.start()
        self.status_message = "🤖 Conversation thread started"
    
    def draw_interface(self, stdscr):
        """Draw the TUI interface"""
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)     # SAM-Alpha
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)    # SAM-Beta
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Ollama-DeepSeek
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # ChatGPT
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Claude
        curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)       # Gemini
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)    # User
        curses.init_pair(8, curses.COLOR_BLUE, curses.COLOR_WHITE)      # System/Status
        
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Header
        header = "🤖 TUI AI Conversation Hub - Multi-Agent Chat"
        stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)
        stdscr.addstr(1, 0, "─" * width)
        
        # Status line
        status_time = datetime.now().strftime("%H:%M:%S")
        status_line = f"📊 Status: {self.status_message} | ⏰ {status_time}"
        stdscr.addstr(2, 0, status_line, curses.color_pair(8))
        stdscr.addstr(3, 0, "─" * width)
        
        # Agent status
        agent_line = "Agents: "
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            color = self.get_agent_color(agent_id)
            agent_line += f"[{agent['name']}] "
        stdscr.addstr(4, 0, agent_line)
        stdscr.addstr(5, 0, "─" * width)
        
        # Conversation area
        conversation_height = height - 9
        conversation_width = width - 4
        
        # Display conversation history
        y = 7
        for msg in list(self.conversation_history)[-conversation_height+2:]:
            if y >= height - 4:
                break
                
            timestamp = msg['timestamp']
            sender = msg['sender']
            message = msg['message']
            agent_id = msg.get('agent_id', 'user')
            
            # Truncate message if too long
            if len(message) > conversation_width:
                message = message[:conversation_width-3] + "..."
            
            # Color code by sender
            color = self.get_agent_color(agent_id)
            
            # Format: [HH:MM:SS] Agent: Message
            line = f"[{timestamp}] {sender}: {message}"
            if len(line) > width - 2:
                line = line[:width-3] + "..."
            
            try:
                stdscr.addstr(y, 2, line, color)
            except:
                pass  # Skip if line doesn't fit
            y += 1
        
        # Typing indicator
        if self.typing_agent:
            agent = self.agents[self.typing_agent]
            typing_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {agent['name']} is typing..."
            color = self.get_agent_color(self.typing_agent)
            try:
                stdscr.addstr(height - 4, 2, typing_msg, color | curses.A_BLINK)
            except:
                pass
        
        # Input line
        stdscr.addstr(height - 3, 0, "─" * width)
        input_line = f"Input: {self.current_input}"
        try:
            stdscr.addstr(height - 2, 0, input_line)
        except:
            pass
        
        # Instructions
        instructions = "Type 'quit' to exit | 'agents' to list agents | 'clear' to clear screen"
        try:
            stdscr.addstr(height - 2, width - len(instructions) - 1, instructions, curses.color_pair(8))
        except:
            pass
        
        stdscr.refresh()
    
    def handle_input(self, stdscr):
        """Handle user input"""
        height, width = stdscr.getmaxyx()
        
        # Get user input
        curses.echo(False)
        curses.curs_set(1)
        
        input_win = curses.newwin(1, width - 7, height - 1, 7)
        input_win.keypad(True)
        
        while self.running:
            try:
                ch = input_win.getch()
                
                if ch == curses.KEY_ENTER or ch == 10:  # Enter key
                    if self.current_input.strip():
                        if self.current_input.strip().lower() == 'quit':
                            self.running = False
                            break
                        elif self.current_input.strip().lower() == 'agents':
                            self.show_agents(stdscr)
                        elif self.current_input.strip().lower() == 'clear':
                            self.conversation_history.clear()
                        else:
                            # Add user message
                            self.add_message("User", self.current_input, 'user')
                            
                            # Generate quick response
                            response = "I understand. Let me think about this..."
                            self.add_message("System", response, 'system')
                        
                        self.current_input = ""
                        input_win.clear()
                        input_win.refresh()
                
                elif ch == curses.KEY_BACKSPACE or ch == 127:
                    if len(self.current_input) > 0:
                        self.current_input = self.current_input[:-1]
                        input_win.clear()
                        input_win.addstr(0, 0, self.current_input)
                        input_win.refresh()
                
                elif ch >= 32 and ch <= 126:  # Printable characters
                    if len(self.current_input) < width - 10:
                        self.current_input += chr(ch)
                        input_win.addch(0, len(self.current_input) - 1, ch)
                        input_win.refresh()
                
            except:
                pass
    
    def show_agents(self, stdscr):
        """Show agent information"""
        height, width = stdscr.getmaxyx()
        
        # Create popup window
        popup_height = len(self.agents) + 4
        popup_width = width - 10
        popup = curses.newwin(popup_height, popup_width, (height - popup_height) // 2, 5)
        popup.box()
        
        popup.addstr(0, (popup_width - 12) // 2, " AI Agents ", curses.A_BOLD)
        
        y = 1
        for agent_id, agent in self.agents.items():
            color = self.get_agent_color(agent_id)
            info = f"{agent['name']} - {agent['specialty']}"
            popup.addstr(y, 2, info, color)
            y += 1
        
        popup.addstr(popup_height - 2, 2, "Press any key to continue...")
        popup.refresh()
        
        popup.getch()
        del popup
    
    def run(self):
        """Run the TUI hub"""
        def main_loop(stdscr):
            # Initialize colors
            curses.start_color()
            curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)     # SAM-Alpha
            curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)    # SAM-Beta
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Ollama-DeepSeek
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # ChatGPT
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Claude
            curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)       # Gemini
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)    # User
            curses.init_pair(8, curses.COLOR_BLUE, curses.COLOR_WHITE)      # System/Status
            
            # Start autonomous conversation
            self.start_autonomous_conversation()
            
            # Main loop with proper refresh
            while self.running:
                try:
                    # Clear and redraw
                    stdscr.clear()
                    height, width = stdscr.getmaxyx()
                    
                    # Header
                    header = "🤖 TUI AI Conversation Hub - Multi-Agent Chat"
                    stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)
                    stdscr.addstr(1, 0, "─" * width)
                    
                    # Status line
                    status_time = datetime.now().strftime("%H:%M:%S")
                    status_line = f"📊 Status: {self.status_message} | ⏰ {status_time}"
                    stdscr.addstr(2, 0, status_line, curses.color_pair(8))
                    stdscr.addstr(3, 0, "─" * width)
                    
                    # Agent status
                    agent_line = "Agents: "
                    for i, (agent_id, agent) in enumerate(self.agents.items()):
                        color = self.get_agent_color(agent_id)
                        agent_line += f"[{agent['name']}] "
                    stdscr.addstr(4, 0, agent_line)
                    stdscr.addstr(5, 0, "─" * width)
                    
                    # Conversation area
                    y = 7
                    for msg in list(self.conversation_history)[-height+10:]:
                        if y >= height - 4:
                            break
                            
                        timestamp = msg['timestamp']
                        sender = msg['sender']
                        message = msg['message']
                        agent_id = msg.get('agent_id', 'user')
                        
                        # Truncate message if too long
                        if len(message) > width - 20:
                            message = message[:width-23] + "..."
                        
                        # Color code by sender
                        color = self.get_agent_color(agent_id)
                        
                        # Format: [HH:MM:SS] Agent: Message
                        line = f"[{timestamp}] {sender}: {message}"
                        if len(line) > width - 2:
                            line = line[:width-3] + "..."
                        
                        try:
                            stdscr.addstr(y, 2, line, color)
                        except:
                            pass  # Skip if line doesn't fit
                        y += 1
                    
                    # Typing indicator
                    if self.typing_agent:
                        agent = self.agents[self.typing_agent]
                        typing_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {agent['name']} is typing..."
                        color = self.get_agent_color(self.typing_agent)
                        try:
                            stdscr.addstr(height - 4, 2, typing_msg, color | curses.A_BLINK)
                        except:
                            pass
                    
                    # Input line
                    stdscr.addstr(height - 3, 0, "─" * width)
                    input_line = f"Input: {self.current_input}"
                    try:
                        stdscr.addstr(height - 2, 0, input_line)
                    except:
                        pass
                    
                    # Instructions
                    instructions = "Type 'quit' to exit | 'agents' to list agents | 'clear' to clear screen"
                    try:
                        stdscr.addstr(height - 2, width - len(instructions) - 1, instructions, curses.color_pair(8))
                    except:
                        pass
                    
                    # Refresh display
                    stdscr.refresh()
                    
                    # Handle input with timeout
                    stdscr.timeout(100)  # 100ms timeout
                    ch = stdscr.getch()
                    
                    if ch != -1:  # Key pressed
                        if ch == curses.KEY_ENTER or ch == 10:  # Enter key
                            if self.current_input.strip():
                                if self.current_input.strip().lower() == 'quit':
                                    self.running = False
                                    break
                                elif self.current_input.strip().lower() == 'agents':
                                    self.show_agents(stdscr)
                                elif self.current_input.strip().lower() == 'clear':
                                    self.conversation_history.clear()
                                else:
                                    # Add user message
                                    self.add_message("User", self.current_input, 'user')
                                    # Quick response
                                    self.add_message("System", "I understand. Let me think about this...", 'system')
                                
                                self.current_input = ""
                        elif ch == curses.KEY_BACKSPACE or ch == 127:
                            if len(self.current_input) > 0:
                                self.current_input = self.current_input[:-1]
                        elif ch >= 32 and ch <= 126:  # Printable characters
                            if len(self.current_input) < width - 20:
                                self.current_input += chr(ch)
                
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    # Handle curses errors gracefully
                    time.sleep(0.1)
                    continue
        
        # Run curses application
        curses.wrapper(main_loop)

if __name__ == "__main__":
    print("🚀 Starting TUI AI Conversation Hub...")
    print("🤖 SAM-MUZE Integration Ready")
    print("💬 Terminal-based multi-agent conversation")
    print("🛑 Press Ctrl+C to exit")
    print("=" * 50)
    
    hub = TUIAIHub()
    
    try:
        hub.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Falling back to simple mode...")
        
        # Simple fallback mode
        print("\n🤖 Simple AI Chat Mode")
        print("Type 'quit' to exit")
        print("-" * 30)
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                
                # Simple response
                responses = [
                    "That's interesting! From my perspective, this relates to SAM-MUZE architecture.",
                    "I see what you mean. Let me analyze this using the Dominant Compression principle.",
                    "Great point! The SAM head model coordinates MUZE submodels for optimal performance.",
                    "I understand. This connects to our work on neural compression and variational inference."
                ]
                
                import random
                response = random.choice(responses)
                print(f"AI: {response}")
                
            except KeyboardInterrupt:
                break
        
        print("👋 Goodbye!")
