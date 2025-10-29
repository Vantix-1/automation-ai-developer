"""
Automation & AI Dev: CLI To-Do List App
Purpose: Manage tasks using a command-line interface (CRUD operations).
Focus: Data structures, persistence, and automation workflow.
"""

tasks = []

def add_task(task):
    tasks.append(task)
    print(f"Task added: {task}")

def view_tasks():
    print("Tasks:")
    for i, task in enumerate(tasks, start=1):
        print(f"{i}. {task}")

def remove_task(index):
    if 0 <= index < len(tasks):
        removed = tasks.pop(index)
        print(f"Removed task: {removed}")
    else:
        print("Invalid index.")

if __name__ == "__main__":
    add_task("Build AI roadmap")
    view_tasks()
