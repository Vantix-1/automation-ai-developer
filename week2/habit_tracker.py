# habit_tracker.py
"""
Automation & AI Dev: Habit Tracker with Persistence
Purpose: Tracks user habits over time and stores progress automatically.
Focus: Data persistence, automation, and foundational AI integration.
"""

habits = {}

def log_habit(habit, status=True):
    habits[habit] = status
    print(f"Habit: {habit} | Completed: {status}")