"""
Automation & AI Dev: Daily Planner Generator
Purpose: Automatically generates a structured daily plan based on tasks.
Focus: Workflow automation and Python loops/dictionaries.
"""

def generate_daily_plan(tasks):
    plan = {}
    for i, task in enumerate(tasks, start=1):
        plan[f"Task {i}"] = task
    return plan

if __name__ == "__main__":
    tasks = ["Check emails", "Write code", "Test AI script", "Plan next task"]
    daily_plan = generate_daily_plan(tasks)
    print("Daily Plan:")
    for k, v in daily_plan.items():
        print(f"{k}: {v}")
