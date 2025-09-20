#!/usr/bin/env python3
"""
Open the enhanced Sentinel dashboard
"""

import webbrowser
import os
from pathlib import Path

def open_dashboard():
    """Open the enhanced dashboard in the browser"""
    dashboard_path = Path("enhanced_dashboard.html").absolute()
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return
    
    print("🚀 Opening Self-Healing Sentinel Dashboard...")
    print("=" * 50)
    print("🌐 Dashboard URL: file://" + str(dashboard_path))
    print("=" * 50)
    print("📋 Features:")
    print("   • Real-time process monitoring")
    print("   • Complete timeline of self-healing process")
    print("   • Detailed fix descriptions")
    print("   • Live metrics and impact assessment")
    print("   • Auto-refresh every 30 seconds")
    print("=" * 50)
    
    # Open in default browser
    webbrowser.open(f"file://{dashboard_path}")
    
    print("✅ Dashboard opened in your browser!")
    print("🔄 The dashboard will auto-refresh every 30 seconds")

if __name__ == "__main__":
    open_dashboard()
