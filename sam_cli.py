#!/usr/bin/env python3
"""
SAM CLI - Interactive Command Line Interface for SAM AGI
Provides terminal-like experience with virtual computer access
"""

import os
import sys
import time
import json
import subprocess
import threading
import shutil
from pathlib import Path
from datetime import datetime
import psutil
import readline  # For better command line editing
import cmd

class SAMTerminal(cmd.Cmd):
    """SAM's Interactive Terminal Interface"""
    
    intro = """
╔══════════════════════════════════════════════════════════╗
║                    🧠 SAM AGI TERMINAL                    ║
║                                                          ║
║  Welcome to SAM's Interactive Command Environment       ║
║  Type 'help' for available commands                      ║
║  Type 'exit' to return to main system                    ║
╚══════════════════════════════════════════════════════════╝

"""
    
    prompt = "sam@agi:~$ "
    
    def __init__(self, sam_system):
        super().__init__()
        self.sam_system = sam_system
        self.current_directory = Path.home()
        self.history_file = Path.home() / ".sam_cli_history"
        self.environment_vars = os.environ.copy()
        
        # Initialize readline with history
        try:
            readline.read_history_file(str(self.history_file))
        except FileNotFoundError:
            pass
        
        # Set up auto-completion
        readline.set_completer(self.complete_path)
        readline.parse_and_bind("tab: complete")
        
        print(f"🎯 SAM Terminal initialized in: {self.current_directory}")
        print(f"💾 System RAM: {psutil.virtual_memory().percent:.1f}% used")
        print(f"🧠 Connected Agents: {len(getattr(self.sam_system, 'connected_agents', {}))}")
    
    def precmd(self, line):
        """Process command before execution"""
        line = line.strip()
        if line:
            # Record command in system metrics
            if hasattr(self.sam_system, 'system_metrics'):
                self.sam_system.system_metrics['cli_commands'] = self.sam_system.system_metrics.get('cli_commands', 0) + 1
            
            # Add to readline history
            readline.add_history(line)
        
        return line
    
    def postcmd(self, stop, line):
        """Process after command execution"""
        # Save history periodically
        if hasattr(readline, 'write_history_file'):
            readline.write_history_file(str(self.history_file))
        
        return stop
    
    def do_help(self, arg):
        """Show available commands"""
        help_text = """
╔══════════════════════════════════════════════════════════╗
║                     🧠 SAM CLI COMMANDS                     ║
╠══════════════════════════════════════════════════════════╣
║                                                                ║
║  📁 FILE SYSTEM COMMANDS:                                       ║
║  • ls, dir          - List directory contents                   ║
║  • cd <path>        - Change directory                           ║
║  • pwd              - Show current directory                    ║
║  • cat <file>       - Display file contents                     ║
║  • mkdir <dir>      - Create directory                          ║
║  • touch <file>     - Create empty file                         ║
║  • cp <src> <dst>   - Copy files/directories                    ║
║  • mv <src> <dst>   - Move/rename files/directories             ║
║  • rm <file>        - Remove files (use -r for directories)     ║
║                                                                ║
║  🤖 SAM AGI COMMANDS:                                           ║
║  • sam <query>      - Ask SAM a question                        ║
║  • agents           - List connected agents                     ║
║  • connect <agent>  - Connect to specific agent                ║
║  • research <topic> - Research using SAM's capabilities        ║
║  • code <task>      - Generate code for tasks                   ║
║  • analyze <file>   - Analyze code or data files                ║
║                                                                ║
║  💻 SYSTEM COMMANDS:                                            ║
║  • status           - Show system status                        ║
║  • processes        - Show running processes                    ║
║  • memory           - Show memory usage                         ║
║  • disk             - Show disk usage                           ║
║  • network          - Show network information                  ║
║                                                                ║
║  🐳 VIRTUAL ENVIRONMENT:                                        ║
║  • docker <cmd>     - Run Docker commands                       ║
║  • python <script>  - Run Python scripts in isolated env        ║
║  • shell <cmd>      - Execute system commands safely            ║
║                                                                ║
║  🔧 UTILITY COMMANDS:                                           ║
║  • clear            - Clear terminal screen                     ║
║  • history          - Show command history                      ║
║  • export <var>=<val> - Set environment variable               ║
║  • env              - Show environment variables                ║
║  • date             - Show current date/time                    ║
║  • whoami           - Show current user                         ║
║                                                                ║
║  🚪 EXIT COMMANDS:                                              ║
║  • exit, quit       - Exit SAM Terminal                         ║
║  • web              - Open web interface                        ║
╚══════════════════════════════════════════════════════════╝
        """
        print(help_text)
    
    def do_ls(self, arg):
        """List directory contents"""
        try:
            path = arg if arg else "."
            items = list(Path(path).iterdir())
            
            if not items:
                print("(empty directory)")
                return
            
            # Sort: directories first, then files
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Display directories in blue
            for item in sorted(dirs):
                size_info = f" ({len(list(item.iterdir()))} items)" if len(list(item.iterdir())) > 0 else ""
                print(f"\033[34m{item.name}/\033[0m{size_info}")
            
            # Display files with sizes
            for item in sorted(files):
                size = item.stat().st_size
                size_str = self._format_size(size)
                print(f"{item.name} ({size_str})")
                
        except Exception as e:
            print(f"Error listing directory: {e}")
    
    def do_cd(self, arg):
        """Change directory"""
        try:
            if not arg:
                new_path = Path.home()
            elif arg == "..":
                new_path = self.current_directory.parent
            elif arg == ".":
                return  # No change
            else:
                new_path = self.current_directory / arg
            
            # Resolve to absolute path
            new_path = new_path.resolve()
            
            if new_path.exists() and new_path.is_dir():
                self.current_directory = new_path
                os.chdir(str(new_path))
                print(f"Changed to: {new_path}")
            else:
                print(f"Directory not found: {new_path}")
                
        except Exception as e:
            print(f"Error changing directory: {e}")
    
    def do_pwd(self, arg):
        """Show current directory"""
        print(self.current_directory)
    
    def do_cat(self, arg):
        """Display file contents"""
        if not arg:
            print("Usage: cat <filename>")
            return
        
        try:
            file_path = self.current_directory / arg
            if file_path.exists() and file_path.is_file():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Limit output for large files
                    if len(content) > 5000:
                        print(content[:5000])
                        print(f"\n... (file truncated, {len(content)} total characters)")
                    else:
                        print(content)
            else:
                print(f"File not found: {arg}")
                
        except Exception as e:
            print(f"Error reading file: {e}")
    
    def do_mkdir(self, arg):
        """Create directory"""
        if not arg:
            print("Usage: mkdir <directory_name>")
            return
        
        try:
            dir_path = self.current_directory / arg
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
    
    def do_touch(self, arg):
        """Create empty file"""
        if not arg:
            print("Usage: touch <filename>")
            return
        
        try:
            file_path = self.current_directory / arg
            file_path.touch()
            print(f"Created file: {file_path}")
        except Exception as e:
            print(f"Error creating file: {e}")
    
    def do_cp(self, arg):
        """Copy files/directories"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: cp <source> <destination>")
            return
        
        try:
            src, dst = args
            src_path = self.current_directory / src
            dst_path = self.current_directory / dst
            
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src} -> {dst}")
            elif src_path.is_dir():
                shutil.copytree(src_path, dst_path)
                print(f"Copied directory: {src} -> {dst}")
            else:
                print(f"Source not found: {src}")
                
        except Exception as e:
            print(f"Error copying: {e}")
    
    def do_mv(self, arg):
        """Move/rename files/directories"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: mv <source> <destination>")
            return
        
        try:
            src, dst = args
            src_path = self.current_directory / src
            dst_path = self.current_directory / dst
            
            shutil.move(str(src_path), str(dst_path))
            print(f"Moved: {src} -> {dst}")
            
        except Exception as e:
            print(f"Error moving: {e}")
    
    def do_rm(self, arg):
        """Remove files/directories"""
        if not arg:
            print("Usage: rm <file> [-r for directories]")
            return
        
        args = arg.split()
        recursive = '-r' in args or '--recursive' in args
        
        if recursive:
            args.remove('-r')
        
        if not args:
            print("Usage: rm <file> [-r for directories]")
            return
        
        target = args[0]
        target_path = self.current_directory / target
        
        try:
            if target_path.is_file():
                target_path.unlink()
                print(f"Removed file: {target}")
            elif target_path.is_dir() and recursive:
                shutil.rmtree(target_path)
                print(f"Removed directory: {target}")
            elif target_path.is_dir():
                print(f"Use 'rm {target} -r' to remove directories")
            else:
                print(f"Target not found: {target}")
                
        except Exception as e:
            print(f"Error removing: {e}")
    
    def do_sam(self, arg):
        """Ask SAM a question"""
        if not arg:
            print("Usage: sam <question>")
            return
        
        try:
            # Use the SAM system to process the query
            context = {"source": "cli", "user": "terminal_user"}
            response = self.sam_system._process_chatbot_message(arg, context)
            print(f"\n🤖 SAM Response:\n{response}\n")
            
        except Exception as e:
            print(f"Error communicating with SAM: {e}")
    
    def do_agents(self, arg):
        """List connected agents"""
        try:
            agents = self.sam_system.agent_configs
            connected = getattr(self.sam_system, 'connected_agents', {})
            
            print("\n🤖 SAM AGENTS:")
            print("=" * 50)
            
            for agent_id, config in agents.items():
                status = "✅ Connected" if agent_id in connected else "⚪ Available"
                specialty = config.get('specialty', 'General AI')
                agent_type = config.get('type', 'Unknown')
                
                print(f"• {config['name']} ({agent_type})")
                print(f"  Specialty: {specialty}")
                print(f"  Status: {status}")
                print()
                
        except Exception as e:
            print(f"Error listing agents: {e}")
    
    def do_connect(self, arg):
        """Connect to specific agent"""
        if not arg:
            print("Usage: connect <agent_id>")
            return
        
        try:
            if arg in self.sam_system.agent_configs:
                config = self.sam_system.agent_configs[arg]
                if arg not in getattr(self.sam_system, 'connected_agents', {}):
                    self.sam_system.connected_agents[arg] = {
                        'config': config,
                        'connected_at': time.time(),
                        'message_count': 0,
                        'muted': False
                    }
                    print(f"✅ Connected to {config['name']}")
                else:
                    print(f"Already connected to {config['name']}")
            else:
                print(f"Agent not found: {arg}")
                
        except Exception as e:
            print(f"Error connecting to agent: {e}")
    
    def do_research(self, arg):
        """Research using SAM's capabilities"""
        if not arg:
            print("Usage: research <topic>")
            return
        
        try:
            # Use web search capabilities
            if hasattr(self.sam_system, 'sam_web_search_available') and self.sam_system.sam_web_search_available:
                from sam_web_search import search_web_with_sam
                results = search_web_with_sam(f"research: {arg}", save_to_drive=False)
                print(f"\n🔍 Research Results for: {arg}")
                print("=" * 50)
                
                if 'results' in results:
                    for i, result in enumerate(results['results'][:5], 1):
                        content = result.get('content', '')[:300]
                        print(f"{i}. {content}...")
                        print()
                else:
                    print("No research results found")
            else:
                # Fallback to basic research
                response = self.sam_system._process_chatbot_message(f"Research this topic: {arg}", {"source": "cli"})
                print(f"\n🔍 Research on: {arg}")
                print("=" * 50)
                print(response)
                
        except Exception as e:
            print(f"Error performing research: {e}")
    
    def do_code(self, arg):
        """Generate code for tasks"""
        if not arg:
            print("Usage: code <task description>")
            return
        
        try:
            # Use SAM's code generation capabilities
            response = self.sam_system._process_chatbot_message(f"Generate code for: {arg}", {"source": "cli"})
            print(f"\n💻 Code Generation for: {arg}")
            print("=" * 50)
            print(response)
            
        except Exception as e:
            print(f"Error generating code: {e}")
    
    def do_analyze(self, arg):
        """Analyze code or data files"""
        if not arg:
            print("Usage: analyze <filename>")
            return
        
        try:
            file_path = self.current_directory / arg
            if file_path.exists():
                # Use SAM's analysis capabilities
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:2000]  # Limit for analysis
                
                response = self.sam_system._process_chatbot_message(
                    f"Analyze this code/file content: {content}", 
                    {"source": "cli", "file": str(file_path)}
                )
                
                print(f"\n🔍 Analysis of: {arg}")
                print("=" * 50)
                print(response)
            else:
                print(f"File not found: {arg}")
                
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    def do_status(self, arg):
        """Show system status"""
        try:
            metrics = self.sam_system.system_metrics
            
            print("\n🖥️  SYSTEM STATUS")
            print("=" * 50)
            print(f"🕒 Uptime: {metrics.get('start_time', 'Unknown')}")
            print(f"🤖 Connected Agents: {len(getattr(self.sam_system, 'connected_agents', {}))}")
            print(f"💬 Total Conversations: {metrics.get('total_conversations', 0)}")
            print(f"🧠 Consciousness Score: {metrics.get('consciousness_score', 0):.2f}")
            print(f"🎓 Learning Events: {metrics.get('learning_events', 0)}")
            print(f"⚡ System Health: {metrics.get('system_health', 'Unknown')}")
            
            # RAM usage
            ram = psutil.virtual_memory()
            print(f"💾 RAM Usage: {ram.percent:.1f}% ({self._format_size(ram.used)} / {self._format_size(ram.total)})")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            print(f"💿 Disk Usage: {disk.percent:.1f}% ({self._format_size(disk.used)} / {self._format_size(disk.total)})")
            
        except Exception as e:
            print(f"Error getting system status: {e}")
    
    def do_processes(self, arg):
        """Show running processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    info = proc.info
                    if info['cpu_percent'] > 0.1 or info['memory_percent'] > 0.1:
                        processes.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            print("\n⚙️  TOP PROCESSES")
            print("=" * 70)
            print("<30")
            print("-" * 70)
            
            for proc in processes[:10]:
                name = proc.get('name', 'Unknown')[:25]
                cpu = proc.get('cpu_percent', 0)
                mem = proc.get('memory_percent', 0)
                pid = proc.get('pid', 0)
                print("<30")
                
        except Exception as e:
            print(f"Error getting processes: {e}")
    
    def do_memory(self, arg):
        """Show memory usage"""
        try:
            mem = psutil.virtual_memory()
            
            print("\n💾 MEMORY USAGE")
            print("=" * 50)
            print(f"Total:     {self._format_size(mem.total)}")
            print(f"Used:      {self._format_size(mem.used)}")
            print(f"Free:      {self._format_size(mem.available)}")
            print(f"Percentage: {mem.percent:.1f}%")
            
            # Swap memory
            swap = psutil.swap_memory()
            print(f"\n🔄 SWAP MEMORY")
            print(f"Total:     {self._format_size(swap.total)}")
            print(f"Used:      {self._format_size(swap.used)}")
            print(f"Free:      {self._format_size(swap.free)}")
            print(f"Percentage: {swap.percent:.1f}%")
            
        except Exception as e:
            print(f"Error getting memory info: {e}")
    
    def do_disk(self, arg):
        """Show disk usage"""
        try:
            partitions = psutil.disk_partitions()
            
            print("\n💿 DISK USAGE")
            print("=" * 50)
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    print(f"\nMount: {partition.mountpoint}")
                    print(f"  Total: {self._format_size(usage.total)}")
                    print(f"  Used:  {self._format_size(usage.used)}")
                    print(f"  Free:  {self._format_size(usage.free)}")
                    print(f"  Usage: {usage.percent:.1f}%")
                except PermissionError:
                    continue
                    
        except Exception as e:
            print(f"Error getting disk info: {e}")
    
    def do_network(self, arg):
        """Show network information"""
        try:
            net_io = psutil.net_io_counters()
            
            print("\n🌐 NETWORK INFORMATION")
            print("=" * 50)
            print(f"Bytes Sent:     {self._format_size(net_io.bytes_sent)}")
            print(f"Bytes Received: {self._format_size(net_io.bytes_recv)}")
            print(f"Packets Sent:   {net_io.packets_sent}")
            print(f"Packets Recv:   {net_io.packets_recv}")
            
            # Network interfaces
            print(f"\n🔌 NETWORK INTERFACES")
            interfaces = psutil.net_if_addrs()
            for name, addrs in interfaces.items():
                print(f"\n{name}:")
                for addr in addrs:
                    if addr.family.name == 'AF_INET':
                        print(f"  IPv4: {addr.address}")
                    elif addr.family.name == 'AF_INET6':
                        print(f"  IPv6: {addr.address}")
                        
        except Exception as e:
            print(f"Error getting network info: {e}")
    
    def do_docker(self, arg):
        """Run Docker commands"""
        if not arg:
            print("Usage: docker <command>")
            print("Example: docker ps, docker run hello-world")
            return
        
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                print("Docker not available on this system")
                return
            
            # Run Docker command safely
            cmd_parts = arg.split()
            result = subprocess.run(['docker'] + cmd_parts, 
                                  capture_output=True, text=True, timeout=30)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Docker command timed out")
        except Exception as e:
            print(f"Error running Docker command: {e}")
    
    def do_python(self, arg):
        """Run Python scripts in isolated environment"""
        if not arg:
            print("Usage: python <script.py> [args...]")
            return
        
        try:
            args = arg.split()
            script_path = self.current_directory / args[0]
            
            if script_path.exists() and script_path.suffix == '.py':
                # Run in subprocess for safety
                result = subprocess.run([sys.executable, str(script_path)] + args[1:], 
                                      capture_output=True, text=True, timeout=30, cwd=str(self.current_directory))
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"Error: {result.stderr}")
            else:
                print(f"Python script not found: {args[0]}")
                
        except subprocess.TimeoutExpired:
            print("Python script execution timed out")
        except Exception as e:
            print(f"Error running Python script: {e}")
    
    def do_shell(self, arg):
        """Execute system commands safely"""
        if not arg:
            print("Usage: shell <command>")
            print("Note: Commands are executed with limited permissions")
            return
        
        # Define allowed commands for safety
        allowed_commands = ['ls', 'pwd', 'echo', 'date', 'whoami', 'uptime', 'df', 'free', 'ps']
        
        cmd_parts = arg.split()
        if cmd_parts[0] not in allowed_commands:
            print(f"Command '{cmd_parts[0]}' not allowed for security reasons")
            print(f"Allowed commands: {', '.join(allowed_commands)}")
            return
        
        try:
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=10)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Command execution timed out")
        except Exception as e:
            print(f"Error executing command: {e}")
    
    def do_clear(self, arg):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def do_history(self, arg):
        """Show command history"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    lines = f.readlines()
                    print("\n📜 COMMAND HISTORY")
                    print("=" * 30)
                    for i, line in enumerate(lines[-20:], len(lines)-19):  # Show last 20
                        print("2d")
            else:
                print("No command history available")
        except Exception as e:
            print(f"Error reading history: {e}")
    
    def do_export(self, arg):
        """Set environment variable"""
        if '=' not in arg:
            print("Usage: export VARIABLE=value")
            return
        
        var, value = arg.split('=', 1)
        self.environment_vars[var] = value
        os.environ[var] = value
        print(f"Set {var}={value}")
    
    def do_env(self, arg):
        """Show environment variables"""
        print("\n🌍 ENVIRONMENT VARIABLES")
        print("=" * 40)
        
        # Show SAM-specific and important vars
        important_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'PYTHONPATH']
        
        for var in important_vars:
            if var in self.environment_vars:
                value = self.environment_vars[var]
                if len(value) > 80:
                    value = value[:77] + "..."
                print("<20")
        
        if arg == '-a':
            print("\n📋 ALL VARIABLES:")
            for var, value in sorted(self.environment_vars.items()):
                if var not in important_vars:
                    if len(value) > 80:
                        value = value[:77] + "..."
                    print("<20")
    
    def do_date(self, arg):
        """Show current date/time"""
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    def do_whoami(self, arg):
        """Show current user"""
        try:
            user = os.getlogin()
            print(f"User: {user}")
        except:
            print(f"User: {os.environ.get('USER', 'unknown')}")
    
    def do_web(self, arg):
        """Open web interface"""
        print("🌐 Opening SAM web interface...")
        print("URL: http://localhost:5004")
        
        try:
            # Try to open browser
            import webbrowser
            webbrowser.open('http://localhost:5004')
            print("✅ Browser opened successfully")
        except:
            print("ℹ️  Please manually open: http://localhost:5004")
    
    def do_exit(self, arg):
        """Exit SAM Terminal"""
        print("\n👋 Exiting SAM Terminal...")
        print("Returning to main SAM system.")
        return True
    
    def do_quit(self, arg):
        """Exit SAM Terminal"""
        return self.do_exit(arg)
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return ".1f"
            size_bytes /= 1024.0
        return ".1f"
    
    def complete_path(self, text, state):
        """Auto-complete file paths"""
        try:
            # Get current directory contents
            current_path = str(self.current_directory)
            if text:
                # Complete partial paths
                path = Path(current_path) / text
                dir_path = path.parent
                prefix = path.name
            else:
                dir_path = Path(current_path)
                prefix = ""
            
            if dir_path.exists() and dir_path.is_dir():
                matches = []
                for item in dir_path.iterdir():
                    if item.name.startswith(prefix):
                        if item.is_dir():
                            matches.append(item.name + '/')
                        else:
                            matches.append(item.name)
                
                if state < len(matches):
                    return matches[state]
                    
        except Exception:
            pass
        
        return None

def launch_sam_terminal(sam_system):
    """Launch the SAM Terminal interface"""
    try:
        terminal = SAMTerminal(sam_system)
        terminal.cmdloop()
    except KeyboardInterrupt:
        print("\n🛑 Terminal interrupted by user")
    except Exception as e:
        print(f"❌ Terminal error: {e}")

if __name__ == "__main__":
    print("🧠 SAM CLI - Interactive Terminal Interface")
    print("Launch this through the main SAM system:")
    print("  from sam_cli import launch_sam_terminal")
    print("  launch_sam_terminal(sam_system)")
