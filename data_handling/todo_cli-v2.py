"""
Automation & AI Dev: Enhanced CLI To-Do App
Purpose: Adds priority levels and deadlines to tasks.
Focus: Task automation, user input, and data handling for AI integration.
"""

import json, os

DATA_FILE = os.path.join(os.path.dirname(__file__), "todo_cli_v2.json")

def load_tasks():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_tasks(tasks):
    with open(DATA_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

# Sample tasks
tasks = load_tasks()
tasks.append({"title": "Practice AI", "completed": False, "priority": "High", "deadline": "2025-11-01"})
save_tasks(tasks)
print("Tasks:", tasks)
