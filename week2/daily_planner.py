"""
Automation & AI Dev: Daily Planner Generator
Purpose: Automatically generates a structured daily plan based on tasks.
Focus: Workflow automation and Python loops/dictionaries.
"""

import json, os

DATA_FILE = os.path.join(os.path.dirname(__file__), "daily_planner.json")

daily_plan = [
    {"time": "08:00", "activity": "Morning Routine"},
    {"time": "09:00", "activity": "Code Practice"},
    {"time": "11:00", "activity": "Automation Scripts"}
]

with open(DATA_FILE, "w") as f:
    json.dump(daily_plan, f, indent=4)

print("Daily plan saved:", daily_plan)
