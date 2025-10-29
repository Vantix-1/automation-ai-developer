"""
Automation & AI Dev: Habit Tracker with Persistence
Purpose: Track and log daily habits with automated storage for long-term analysis.
Future AI Integration: Generate productivity insights or reminders based on habit trends.
"""

import json
import os
from datetime import date

DATA_FILE = "habit_data.json"

# Load existing habits or initialize empty
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        habits = json.load(f)
else:
    habits = {}

def log_habit(habit_name, completed=True):
    today = str(date.today())
    if today not in habits:
        habits[today] = {}
    habits[today][habit_name] = completed
    save_habits()
    status = "✅" if completed else "❌"
    print(f"{today} | {habit_name}: {status}")

def save_habits():
    with open(DATA_FILE, "w") as f:
        json.dump(habits, f, indent=4)

def show_habits():
    if not habits:
        print("No habits logged yet.")
        return
    for day, day_habits in sorted(habits.items()):
        print(f"\nDate: {day}")
        for habit, completed in day_habits.items():
            status = "✅" if completed else "❌"
            print(f" - {habit}: {status}")

if __name__ == "__main__":
    log_habit("Exercise")
    log_habit("Code 1 hour")
    show_habits()
