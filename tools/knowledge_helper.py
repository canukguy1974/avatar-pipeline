import subprocess
import json
import os
import sys

# Mapping of file paths to KNOWLEDGE.md sections
SECTION_MAPPING = {
    "web/": "1. Frontend (/web)",
    "src/app/main.py": "2. Backend (/src/app) or 3. Audio Pipeline (TTS)",
    "src/app/eleven_tts_ws.py": "3. Audio Pipeline (TTS)",
    "src/app/hls_writer.py": "2. Backend (/src/app) or 📼 HLS Streaming Mechanics",
    "InfiniteTalk/workers/video/": "4. Video Worker (/InfiniteTalk/workers/video)",
    ".env": "⚠️ Known Gotchas & Patterns",
    "docker-compose.yml": "🏗️ Core Architecture",
    "KNOWLEDGE.md": "Knowledge Base Maintenance",
    ".agent/": "AI Agent Workflows",
}

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def get_recent_changes():
    print("--- Scanning for Recent Changes ---")
    
    # 1. Check git status
    status = run_command("git status --short")
    if status:
        print("\n[Uncommitted/Staged Changes]:")
        print(status)
    else:
        print("\nNo uncommitted changes.")

    # 2. Check last 3 commits
    log = run_command("git log -n 3 --name-only --pretty=format:'%h - %s'")
    print("\n[Last 3 Commits]:")
    print(log)
    
    return status + "\n" + log

def suggest_updates(changes):
    suggestions = set()
    for path, section in SECTION_MAPPING.items():
        if path in changes:
            suggestions.add(section)
    
    if suggestions:
        print("\n--- Knowledge Base Suggestions ---")
        print("Based on recent changes, please review and update these sections in KNOWLEDGE.md:")
        for s in suggestions:
            print(f" - {s}")
    else:
        print("\nNo specific Knowledge Base updates suggested based on file paths.")
        print(f"(Debug: Change string length: {len(changes)})")

if __name__ == "__main__":
    changes = get_recent_changes()
    suggest_updates(changes)
