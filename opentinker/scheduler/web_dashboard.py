#!/usr/bin/env python3
"""
Web Dashboard Server for OpenTinker

Serves the HTML dashboard and provides a simple HTTP server.
"""

import argparse
import http.server
import socketserver
import os
from pathlib import Path


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS enabled"""
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS preflight"""
        self.send_response(200)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description='OpenTinker Web Dashboard Server')
    parser.add_argument('--port', type=int, default=8081, help='Port to run the server on (default: 8080)')
    parser.add_argument('--scheduler-url', default='http://localhost:8767', 
                       help='Scheduler API URL (default: http://localhost:8767)')
    args = parser.parse_args()
    
    # Change to the directory containing the HTML file
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    print("="*70)
    print("🌐 OpenTinker Web Dashboard")
    print("="*70)
    print(f"\n📍 Dashboard URL: http://localhost:{args.port}")
    print(f"🔗 Scheduler URL: {args.scheduler_url}")
    print(f"📁 Serving from: {dashboard_dir}")
    print("\n💡 Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    # Start server
    with socketserver.TCPServer(("", args.port), CORSRequestHandler) as httpd:
        try:
            print(f"✅ Server running on port {args.port}")
            print(f"\n🚀 Open http://localhost:{args.port}/web_dashboard.html in your browser\n")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down server...")


if __name__ == '__main__':
    main()
