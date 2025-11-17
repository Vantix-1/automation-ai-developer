"""
Automation & AI Dev: CLI To-Do List App
Purpose: Manage tasks using a command-line interface (CRUD operations).
Focus: Data structures, persistence, and automation workflow.
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

tasks = load_tasks()

# Example: add a task
tasks.append({"title": "Learn Flask", "completed": False})
save_tasks(tasks)
print("Tasks:", tasks)
