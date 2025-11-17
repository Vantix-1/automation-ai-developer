"""
Automation & AI Dev: AI Productivity Tip Bot
Purpose: Provides AI-generated productivity tips to the user.
Focus: Early AI integration for task suggestions and insights.
"""

import json, os, random

DATA_FILE = os.path.join(os.path.dirname(__file__), "tips.json")

tips = [
    "Focus on one task at a time",
    "Take breaks every hour",
    "Plan tomorrow tonight",
    "Automate repetitive tasks"
]

tip = random.choice(tips)
print("Tip of the day:", tip)

with open(DATA_FILE, "w") as f:
    json.dump({"tip_of_the_day": tip}, f, indent=4)
