#!/usr/bin/env python3
"""
SAM Terminal Routes - Web interface integration for SAM CLI
"""

def add_terminal_routes(app, sam_system):
    """Add terminal routes to Flask app"""
    
    @app.route('/terminal')
    def terminal_interface():
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 SAM Terminal</title>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; margin: 0; padding: 20px; }
        .terminal { background: #111; border: 1px solid #0f0; padding: 20px; border-radius: 5px; }
        .output { white-space: pre-wrap; margin-bottom: 10px; max-height: 400px; overflow-y: auto; }
        .input-line { display: flex; }
        .prompt { color: #0f0; margin-right: 10px; }
        .input { background: transparent; border: none; color: #0f0; flex: 1; outline: none; font-family: monospace; }
    </style>
</head>
<body>
    <div class="terminal">
        <div class="output" id="output">🧠 SAM AGI Terminal\nType 'help' for available commands or 'exit' to quit.\n\n</div>
        <div class="input-line">
            <span class="prompt">sam@agi:~$</span>
            <input type="text" class="input" id="command-input" autofocus>
        </div>
    </div>

    <script>
        const output = document.getElementById('output');
        const input = document.getElementById('command-input');
        let commandHistory = [];
        let historyIndex = -1;

        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                const command = input.value.trim();
                if (command) {
                    commandHistory.push(command);
                    historyIndex = commandHistory.length;
                    executeCommand(command);
                }
                input.value = '';
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (historyIndex > 0) {
                    historyIndex--;
                    input.value = commandHistory[historyIndex];
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (historyIndex < commandHistory.length - 1) {
                    historyIndex++;
                    input.value = commandHistory[historyIndex];
                } else {
                    historyIndex = commandHistory.length;
                    input.value = '';
                }
            }
        });

        async function executeCommand(cmd) {
            output.textContent += `sam@agi:~$ ${cmd}\n`;
            input.disabled = true;

            try {
                const response = await fetch('/api/terminal/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: cmd })
                });

                const result = await response.json();
                output.textContent += result.output + '\n\n';
            } catch (error) {
                output.textContent += `Error: ${error.message}\n\n`;
            }

            input.disabled = false;
            input.focus();
            output.scrollTop = output.scrollHeight;
        }
    </script>
</body>
</html>
        '''
    
    @app.route('/api/terminal/execute', methods=['POST'])
    def execute_terminal_command():
        try:
            data = request.get_json()
            command = data.get('command', '')
            
            # Execute command through SAM CLI
            if command == 'help':
                result = """🧠 SAM CLI COMMANDS:
• help - Show this help
• ls - List directory
• sam <query> - Ask SAM
• exit - Exit terminal"""
            elif command == 'ls':
                import os
                result = '\n'.join(os.listdir('.'))
            elif command.startswith('sam ') and len(command) > 4:
                query = command[4:]
                result = sam_system._process_chatbot_message(query, {"source": "terminal"})
            elif command == 'exit':
                result = 'Exiting terminal...'
            else:
                result = f"Command not implemented: {command}"
            
            return jsonify({"output": result})
        except Exception as e:
            return jsonify({"output": f"Error: {str(e)}"}), 500

def add_terminal_commands(sam_system):
    """Add terminal commands to the slash command handler"""
    
    # This function should be called from the _process_slash_command method
    def handle_terminal_command(cmd, args, sam_instance):
        if cmd in ['/terminal', '/cli']:
            print('🧠 Launching SAM Terminal...')
            # Launch terminal in background thread
            import threading
            from sam_cli import launch_sam_terminal
            terminal_thread = threading.Thread(target=launch_sam_terminal, args=(sam_instance,), daemon=True)
            terminal_thread.start()
            return 'SAM Terminal launched! Use commands like ls, cd, sam <query>, research <topic>, etc.'
        return None
    
    return handle_terminal_command
