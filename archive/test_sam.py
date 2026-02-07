#!/usr/bin/env python3
print("Starting SAM Hub test...")

try:
    from flask import Flask
    print("Flask imported successfully")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return "SAM Hub Test Working!"
    
    print("About to start Flask app...")
    app.run(host='127.0.0.1', port=8086, debug=False)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
