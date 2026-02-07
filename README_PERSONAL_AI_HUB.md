# Personal AI Conversation Hub - SAM 2.0

🏠 **Your Private AGI Multi-Agent Conversation System**

> **⚠️ Note**: This is the original conversation hub. The complete SAM 2.0 AGI system with latent-space morphogenesis, dominant compression, and neural core integration is documented in [README.md](./README.md).
> 
> For the research paper and full technical documentation, see the main README.

---

## 🚀 Quick Start (SAM 2.0 - Recommended)

```bash
cd /Users/samueldasari/Personal/NN_C

# 1. Build the neural core
make shared

# 2. Run the SAM 2.0 hub
python3 correct_sam_hub.py

# 3. Open browser
curl http://127.0.0.1:8080
```

---

## 🤖 What's New in SAM 2.0

### 🧠 AGI-Style Architecture
- **Latent-Space Morphogenesis**: Dynamic concept birth and network expansion
- **Dominant Compression**: Unified optimization (J - βH - λC + ηI)
- **Clone-Based Submodels**: Task specialization via transfusion/distillation
- **Verification-Grounded Knowledge**: Search→Augment→Relay→Verify→Save pipeline
- **Self-Preserving Identity**: Continuous learning without catastrophic forgetting
- **Consciousness Loss**: Self-modeling via L_cons = KL(World || Self-Prediction)
- **Introspective Agency**: Avoids actions with low self-model confidence

### 🔬 Test & Validate
```bash
# Run AGI growth experiments
python3 agi_test_framework.py

# Results saved to: AGI_TEST_RESULTS/
```

---

## 📚 Full Documentation

**[README.md](./README.md)** - Complete technical specification with:
- Mathematical foundations
- AGI formal definition
- Implementation details
- Experimental results
- API reference

---

**Version**: 2.0 - Full Context Morphogenesis  
**Status**: ✅ Complete & Operational  
**License**: MIT


🏠 **Your Private Multi-Agent Conversation System**

## 🚀 Quick Start

### Option 1: Easy Launcher (Recommended)
```bash
cd /Users/samueldasari/Personal/NN_C
python3 start_personal_ai_hub.py
```

### Option 2: Direct Start
```bash
cd /Users/samueldasari/Personal/NN_C
python3 personal_ai_conversation_hub.py
```

Then open: http://127.0.0.1:8080

## 🤖 Available AI Agents

### ✅ Available Out of the Box
- **SAM-Alpha**: Research & Analysis (Local SAM Neural Network)
- **SAM-Beta**: Synthesis & Application (Local SAM Neural Network)
- **Ollama-Llama2**: General Conversation (Local Ollama)
- **Ollama-DeepSeek**: Technical Analysis & Coding (Local Ollama)

### 🔑 API Key Required (Optional)
- **Claude-3.5-Sonnet**: Advanced Reasoning & Analysis
- **Claude-3-Haiku**: Fast Conversation & Tasks
- **Gemini-Pro**: Multimodal Understanding
- **GPT-4**: General Intelligence & Problem Solving
- **GPT-3.5-Turbo**: Fast Conversation & Assistance
- **HuggingFace Models**: Custom Models (Configurable)

## 🔑 Setting Up API Keys

### Claude (Anthropic)
```bash
export ANTHROPIC_API_KEY='your-claude-api-key'
```

### Gemini (Google)
```bash
export GOOGLE_API_KEY='your-gemini-api-key'
```

### OpenAI (GPT)
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### HuggingFace
```bash
export HUGGINGFACE_API_KEY='your-huggingface-key'
```

## 💬 How to Use

1. **Start the Hub**: Run the launcher or direct start command
2. **Open Browser**: Go to http://127.0.0.1:8080
3. **Connect Agents**: Click "Connect" on agents in the sidebar
4. **Send Messages**: 
   - Select an agent and type a message
   - Click "Send" to message that specific agent
   - Click "Broadcast to All" to message all connected agents
5. **Conversations**: All conversations are saved automatically

## 🎯 Features

### 🔐 Private & Secure
- Only you can access your conversation hub
- All conversations are local and private
- No data sharing with third parties

### 🤖 Multiple AI Providers
- **Local Models**: SAM, Ollama (offline, private)
- **Cloud Models**: Claude, Gemini, GPT (online, powerful)
- **Custom Models**: HuggingFace integration

### 💬 Rich Conversations
- Each agent has unique personality and specialty
- Agents can have different capabilities
- Real-time multi-agent conversations
- Conversation history saved

### 🌐 Web Access
- Agents can access current information
- Self-RAG capabilities for local agents
- Knowledge integration across conversations

## 📁 File Structure

```
/Users/samueldasari/Personal/NN_C/
├── personal_ai_conversation_hub.py    # Main hub application
├── start_personal_ai_hub.py          # Easy launcher
├── README_PERSONAL_AI_HUB.md         # This file
└── personal_ai_conversation_*.json   # Saved conversations
```

## 🛠️ Technical Details

### Architecture
- **Backend**: Flask + SocketIO (Python)
- **Frontend**: HTML5 + JavaScript + CSS
- **Real-time**: WebSocket connections
- **Storage**: Local JSON files

### Agent Types
- **Local**: SAM Neural Networks, Ollama models
- **API**: Claude, Gemini, GPT via official APIs
- **Custom**: HuggingFace model integration

### Security
- Local-only access by default
- API keys stored in environment variables
- No external data sharing
- Conversation history saved locally

## 🚀 Advanced Usage

### Adding Custom Agents
You can extend the system by adding new agent configurations in the `initialize_agent_configs()` method.

### Custom Prompts
Each agent can be customized with different system prompts and personalities.

### Integration
The hub can be integrated with other tools and services through the API endpoints.

## 🆘 Troubleshooting

### Port Already in Use
```bash
lsof -ti:8080 | xargs kill -9
```

### Dependencies Missing
```bash
pip3 install --break-system-packages flask flask-socketio eventlet
```

### Ollama Not Working
```bash
# Make sure Ollama is running
ollama serve

# Check available models
ollama list
```

### API Keys Not Working
- Check that API keys are correctly set
- Verify API key permissions
- Check internet connection for cloud models

## 📞 Support

This is a personal, private AI conversation hub. All conversations and data remain on your local machine.

**Created by**: Your Personal AI Assistant
**Version**: 1.0
**Last Updated**: 2025-02-05
