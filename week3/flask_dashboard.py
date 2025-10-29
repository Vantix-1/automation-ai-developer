# flask_dashboard.py
from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

# Path to Week 2 JSON data
DATA_DIR = os.path.join(os.path.dirname(__file__), "../week2/")

def load_json(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

@app.route("/")
def dashboard():
    # Load JSON data
    tasks = load_json("todo_cli_v2.json")
    habits = load_json("habit_tracker.json")
    daily_plan = load_json("daily_planner.json")
    
    # Example summary stats
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.get("completed"))
    total_habits = len(habits)
    completed_habits = sum(1 for h in habits if h.get("completed"))

    return render_template(
        "dashboard.html",
        tasks=tasks,
        habits=habits,
        daily_plan=daily_plan,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        total_habits=total_habits,
        completed_habits=completed_habits
    )

@app.route("/api/tasks")
def api_tasks():
    tasks = load_json("todo_cli_v2.json")
    return jsonify(tasks)

@app.route("/api/habits")
def api_habits():
    habits = load_json("habit_tracker.json")
    return jsonify(habits)

if __name__ == "__main__":
    app.run(debug=True)
