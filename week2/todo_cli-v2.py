"""
Automation & AI Dev: Enhanced CLI To-Do App
Purpose: Adds priority levels and deadlines to tasks.
Focus: Task automation, user input, and data handling for AI integration.
"""

tasks = []

def add_task(task, priority="Medium", deadline=None):
    tasks.append({"task": task, "priority": priority, "deadline": deadline})
    print(f"Task added: {task} | Priority: {priority} | Deadline: {deadline}")

def view_tasks():
    print("Tasks:")
    for i, task in enumerate(tasks, start=1):
        print(f"{i}. {task['task']} | Priority: {task['priority']} | Deadline: {task['deadline']}")

if __name__ == "__main__":
    add_task("Build AI roadmap", "High", "2025-11-01")
    view_tasks()
