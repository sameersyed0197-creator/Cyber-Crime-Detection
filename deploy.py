#!/usr/bin/env python3
"""
Simple Deployment Script for CyberCrime Detection Project
Handles browser compatibility, version issues, and server setup
"""

import subprocess
import sys
import os
import platform
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_simple_frontend():
    """Create a simple, compatible frontend"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberGuard AI - Simple Deploy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white; 
            padding: 20px; 
        }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .card { 
            background: #16213e; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            border: 1px solid #0f3460;
        }
        .btn { 
            background: #0f3460; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px;
        }
        .btn:hover { background: #1e5f8b; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #2d5a27; }
        .error { background: #5a2727; }
        .loading { background: #5a4527; }
        #results { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ CyberGuard AI</h1>
            <p>Simple Deployment Interface</p>
        </div>

        <div class="card">
            <h3>🚀 Quick Start</h3>
            <p>Upload a CSV file and run analysis</p>
            <input type="file" id="fileInput" accept=".csv" style="margin: 10px 0;">
            <br>
            <button class="btn" onclick="uploadFile()">📤 Upload Dataset</button>
            <button class="btn" onclick="runAnalysis()">🔍 Run Analysis</button>
        </div>

        <div id="status" class="hidden"></div>
        <div id="results" class="hidden"></div>
    </div>

    <script>
        const API_BASE = 'http://localhost:8000';
        let token = null;

        // Auto-login for demo
        async function autoLogin() {
            try {
                const response = await fetch(`${API_BASE}/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: 'demo@example.com',
                        password: 'demo123'
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    token = data.access_token;
                    showStatus('✅ Connected to server', 'success');
                } else {
                    // Create demo user if doesn't exist
                    await createDemoUser();
                }
            } catch (error) {
                showStatus('❌ Server not running. Please start the server first.', 'error');
            }
        }

        async function createDemoUser() {
            try {
                const response = await fetch(`${API_BASE}/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: 'demo@example.com',
                        username: 'demo',
                        password: 'demo123'
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    token = data.access_token;
                    showStatus('✅ Demo user created and connected', 'success');
                }
            } catch (error) {
                showStatus('❌ Failed to create demo user', 'error');
            }
        }

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showStatus('❌ Please select a CSV file', 'error');
                return;
            }

            if (!token) {
                showStatus('❌ Not connected to server', 'error');
                return;
            }

            showStatus('📤 Uploading file...', 'loading');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`${API_BASE}/upload-dataset`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` },
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    showStatus(`✅ File uploaded: ${data.rows} rows, ${data.columns.length} columns`, 'success');
                } else {
                    showStatus('❌ Upload failed', 'error');
                }
            } catch (error) {
                showStatus('❌ Upload error: ' + error.message, 'error');
            }
        }

        async function runAnalysis() {
            if (!token) {
                showStatus('❌ Not connected to server', 'error');
                return;
            }

            showStatus('🔍 Running analysis...', 'loading');

            try {
                const response = await fetch(`${API_BASE}/full-analysis`, {
                    method: 'POST',
                    headers: { 
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });

                if (response.ok) {
                    const data = await response.json();
                    showResults(data);
                } else {
                    showStatus('❌ Analysis failed', 'error');
                }
            } catch (error) {
                showStatus('❌ Analysis error: ' + error.message, 'error');
            }
        }

        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.className = `status ${type}`;
            statusDiv.textContent = message;
            statusDiv.classList.remove('hidden');
        }

        function showResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="card">
                    <h3>📊 Analysis Results</h3>
                    <p><strong>Total Samples:</strong> ${data.total_samples}</p>
                    <p><strong>Safe Traffic:</strong> ${data.safe_count}</p>
                    <p><strong>Threat Traffic:</strong> ${data.threat_count}</p>
                    <p><strong>Detection Rate:</strong> ${data.detection_rate}</p>
                    <p><strong>DDDQN Accuracy:</strong> ${(data.model_stats.dddqn_accuracy * 100).toFixed(1)}%</p>
                    <p><strong>Random Forest Accuracy:</strong> ${(data.model_stats.rf_accuracy * 100).toFixed(1)}%</p>
                </div>
            `;
            resultsDiv.classList.remove('hidden');
            showStatus('✅ Analysis complete!', 'success');
        }

        // Auto-connect on page load
        window.onload = autoLogin;
    </script>
</body>
</html>'''
    
    with open('simple_frontend.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ Simple frontend created: simple_frontend.html")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting server...")
    try:
        # Start server in background
        if platform.system() == "Windows":
            subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
        else:
            subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
        
        print("✅ Server starting at http://localhost:8000")
        print("✅ API docs at http://localhost:8000/docs")
        return True
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def open_browser():
    """Open browser with the application"""
    try:
        frontend_path = Path("simple_frontend.html").absolute()
        webbrowser.open(f"file://{frontend_path}")
        print("✅ Browser opened")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")

def main():
    """Main deployment function"""
    print("🛡️ CyberGuard AI - Simple Deployment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create simple frontend
    create_simple_frontend()
    
    # Start server
    if start_server():
        print("\n⏳ Waiting for server to start...")
        import time
        time.sleep(3)
        
        # Open browser
        open_browser()
        
        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("📱 Frontend: simple_frontend.html")
        print("🌐 Server: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("\n💡 Instructions:")
        print("1. Upload a CSV file with cybercrime data")
        print("2. Click 'Run Analysis' to get results")
        print("3. Server runs on port 8000")
        print("\n🔧 Troubleshooting:")
        print("- If browser doesn't open, manually open simple_frontend.html")
        print("- If server fails, check if port 8000 is available")
        print("- For issues, check the console output")
        
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()