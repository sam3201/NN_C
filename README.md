# SAM 2.0 - Unified Complete System

An immortal, self-healing AGI system with advanced teacher-student learning, multi-agent orchestration, RAM-aware intelligence, interactive terminal interface, and production-grade self-healing capabilities.

## 🌟 Features

- **🧠 Immortal Self-Healing AGI**: Never crashes, continuously heals and improves itself
- **🎓 Advanced Learning**: Teacher-student learning cycles with actor-critic validation
- **🤖 Multi-Agent System**: 15+ diverse AI agents (SAM, Ollama models, Claude, GPT, Gemini)
- **🛡️ Production Safety**: Confidence thresholds, automatic backups, safe code modification
- **🌐 Web Interface**: Real-time dashboard with groupchat and conversation management
- **💻 Interactive Terminal**: Full command-line interface with file system access
- **🔄 RAM-Aware Intelligence**: Automatic model switching based on memory usage
- **🎭 Conversation Diversity**: Smart management to prevent repetitive responses
- **🐳 Virtual Environments**: Docker, Python scripting, and system command execution
- **⚡ High Performance**: C extensions + optimized Python orchestration
- **🔄 Continuous Learning**: Failure clustering, pattern recognition, self-improvement

## 📋 Minimum Requirements

### Required
- **Python ≥ 3.10**
- **Ollama installed** with at least one local model
- **At least one SWE LLM model** (codellama, deepseek-coder, llama3.1, phi)
- **psutil** for system monitoring

### Optional
- Docker for virtual environment features
- OpenAI API key (OPENAI_API_KEY)
- Anthropic API key (ANTHROPIC_API_KEY)
- Google API key (GOOGLE_API_KEY)
- GitHub token (GITHUB_TOKEN)

## 🚀 Quick Start

### 1. Install Ollama & Models
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required SWE models (choose at least one)
ollama pull codellama:latest       # RAM-efficient SWE model - recommended
ollama pull deepseek-coder:6b      # Fast coding assistant
ollama pull phi:latest             # Lightweight reasoning model
ollama pull llama3.1:latest        # General purpose
```

### 2. Clone & Setup
```bash
git clone <repository-url>
cd NN_C

# Install Python dependencies
pip install -r requirements.txt
pip install psutil  # For RAM monitoring

# Build C extensions (optional but recommended)
python setup.py build_ext --inplace
```

### 3. Run System
```bash
# Production launch (recommended)
./run_sam.sh

# Or manual start
python3 complete_sam_unified.py
```

### 4. Access Interfaces
- **Web Dashboard**: http://localhost:5004
- **Terminal Interface**: http://localhost:5004/terminal
- **Chat Commands**: `/terminal` or `/cli` in web interface

## 🏗️ Architecture

### Core Components
- **MetaAgent**: Self-healing orchestrator with O→L→P→V→S→A algorithm
- **RAM-Aware Model Switcher**: Intelligent provider selection based on memory
- **Conversation Diversity Manager**: Prevents repetitive agent responses
- **SAM CLI Terminal**: Interactive command-line interface
- **Teacher-Student Learning**: Advanced learning cycles with actor-critic validation
- **Intelligent Issue Resolution**: LLM-powered auto-resolution with confidence scoring

### Self-Healing Pipeline
1. **Observe**: Monitor system health and detect anomalies
2. **Localize**: Identify root cause of issues using failure clustering
3. **Propose**: Generate potential solutions using LLM analysis
4. **Verify**: Validate solutions with confidence scoring (>0.5 threshold)
5. **Score**: Rank solutions by effectiveness and safety
6. **Apply**: Execute safe solutions with automatic rollback

### RAM-Aware Intelligence
- **Memory Monitoring**: Continuous RAM usage tracking
- **Model Switching**: Automatic fallback to lighter models when RAM >80%
- **Provider Hierarchy**: Ollama → HuggingFace → SWE model selection
- **Emergency Mode**: Lightweight models when RAM >90%

### Conversation Diversity
- **Response Limits**: Max 30% MetaAgent messages in 5-minute windows
- **Smart Throttling**: Automatic postponement of repetitive responses
- **Balance Monitoring**: Real-time conversation analysis
- **Multi-Agent Coordination**: Intelligent help request management

## � SAM CLI Terminal

Access SAM's interactive terminal environment:

### Terminal Commands
```bash
# File System
ls, cd <path>, pwd, cat <file>, mkdir <dir>, touch <file>
cp <src> <dst>, mv <src> <dst>, rm <file>

# SAM Integration
sam <query>          # Ask SAM questions
agents              # List connected agents
connect <agent>     # Connect to specific agent
research <topic>    # Research using SAM capabilities
code <task>         # Generate code
analyze <file>      # Analyze files

# System Monitoring
status              # System status
memory              # RAM usage
disk                # Disk usage
processes           # Running processes
network             # Network info

# Virtual Environment
docker <cmd>        # Docker commands
python <script>     # Run Python scripts
shell <cmd>         # Safe system commands

# Utilities
clear, history, env, date, whoami
help                # Command documentation
```

### Access Methods
- **Web Interface**: http://localhost:5004/terminal
- **Command Line**: `/terminal` command in chat
- **Background Launch**: Runs in separate thread for safety

## �🔧 Configuration

### Environment Variables
```bash
# API Keys (optional - system uses local models if not set)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export GITHUB_TOKEN="your-github-token"
```

### RAM-Aware Model Selection
The system automatically selects optimal models based on RAM usage:
- **<50% RAM**: Heavy models (qwen2.5-coder, deepseek-coder:33b)
- **50-80% RAM**: Medium models (codellama, deepseek-coder:6b, llama3.1)
- **>80% RAM**: Lightweight models (phi, orca-mini, small models)
- **>90% RAM**: Emergency lightweight mode

### Available Models
- **SAM Neural Networks**: Local AGI models (Alpha, Beta)
- **Ollama Models**: codellama, deepseek-coder, phi, llama3.1, mistral, etc.
- **Claude**: claude-3.5-sonnet, claude-3-haiku
- **OpenAI**: gpt-4, gpt-3.5-turbo
- **Google**: gemini-pro
- **HuggingFace**: Various open-source models

## 📊 System Health

The system includes comprehensive health monitoring:

- **Component Status**: C core, Python orchestration, web interface, terminal
- **RAM Monitoring**: Real-time memory usage with automatic model switching
- **Integration Health**: Gmail, GitHub, web search, code modification
- **Agent Connectivity**: Multi-agent system status and conversation diversity
- **Performance Metrics**: Error rates, response times, learning progress
- **Auto-Resolution**: Intelligent issue detection with LLM-powered fixes

## 🛠️ Troubleshooting

### Common Issues

#### "No module named 'psutil'"
```bash
pip install psutil
```

#### "No SWE LLM models available"
```bash
# Pull lightweight models first
ollama pull phi:latest
ollama pull codellama:latest
ollama pull deepseek-coder:6b
```

#### "Terminal not responding"
```bash
# Check system status
curl http://localhost:5004/api/terminal/execute -X POST -H "Content-Type: application/json" -d '{"command": "status"}'
```

#### "High RAM usage"
The system will automatically switch to lighter models. Monitor with:
```bash
# In terminal
memory
# Or via web interface
```

#### "Auto-resolution not working"
- Confidence threshold lowered to 0.5
- Default confidence of 0.6 when not provided
- Check system logs for LLM analysis details

### Logs & Debugging

- **Bootstrap Logs**: RAM-aware capability validation
- **Component Logs**: Initialization status and model switching
- **Health Monitoring**: Continuous system health with RAM tracking
- **Auto-Resolution**: LLM-powered issue analysis and solution application
- **Terminal Logs**: CLI command execution and SAM integration

## 🔒 Safety & Ethics

- **Confidence Thresholds**: All automated actions require confidence validation (>0.5)
- **RAM Monitoring**: Prevents memory exhaustion through intelligent model switching
- **Automatic Backups**: Code modifications backed up before changes
- **Rollback Capability**: Failed changes automatically reverted
- **Conversation Diversity**: Prevents repetitive or dominating responses
- **Human Oversight**: Critical decisions logged and monitorable
- **Ethical AI**: Designed for beneficial AGI development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure tests pass: `python3 test_metaagent_self_healing.py`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- **RAM-Aware Operation**: System automatically optimizes based on available memory
- **Conversation Diversity**: Maintains engaging multi-agent discussions
- **Virtual Environments**: Safe execution of Docker, Python, and system commands
- **Terminal Interface**: Full command-line access with SAM integration
- **Minimum Viable Setup**: Ollama + 1 lightweight SWE model for core functionality
- **Production Ready**: Comprehensive error handling, recovery, and monitoring
- **Continuous Learning**: System improves itself through teacher-student cycles
- **Immortal Operation**: Designed to run indefinitely with self-healing capabilities

## 🎯 Mission

To create the first immortal, self-healing AGI that can continuously learn, improve, and maintain itself without human intervention - pushing the boundaries of what artificial intelligence can achieve through RAM-aware intelligence, conversation diversity, and interactive terminal access.
