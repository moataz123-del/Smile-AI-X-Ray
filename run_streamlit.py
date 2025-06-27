#!/usr/bin/env python3
"""
Script to run the Streamlit Dental X-Ray AI Detection application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("🚀 Starting Dental X-Ray AI Detection Application...")
        print("📱 The application will open in your default web browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
    except ImportError:
        print("❌ Streamlit is not installed. Please install it first:")
        print("pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 