"""
Automation & AI Dev: Habit Tracker with Persistence
Purpose: Track and log daily habits with automated storage for long-term analysis.
Future AI Integration: Generate productivity insights or reminders based on habit trends.
"""
import json, os

DATA_FILE = os.path.join(os.path.dirname(__file__), "habit_tracker.json")

habits = [
    {"name": "Exercise", "completed": True},
    {"name": "Read", "completed": False},
    {"name": "Code Practice", "completed": True}
]

with open(DATA_FILE, "w") as f:
    json.dump(habits, f, indent=4)

print("Habit tracker updated:", habits)
