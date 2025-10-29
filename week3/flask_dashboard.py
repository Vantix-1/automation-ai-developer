from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../week2/")

def load_json(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

@app.route("/")
def dashboard():
    tasks = load_json("todo_cli_v2.json")
    habits = load_json("habit_tracker.json")
    daily_plan = load_json("daily_planner.json")
    weekly_report = load_json("weekly_report.json")
    tips = load_json("tips.json")

    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.get("completed"))
    total_habits = len(habits)
    completed_habits = sum(1 for h in habits if h.get("completed"))

    return render_template(
        "dashboard.html",
        tasks=tasks,
        habits=habits,
        daily_plan=daily_plan,
        weekly_report=weekly_report,
        tips=tips,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        total_habits=total_habits,
        completed_habits=completed_habits
    )

@app.route("/api/tasks")
def api_tasks():
    return jsonify(load_json("todo_cli_v2.json"))

@app.route("/api/habits")
def api_habits():
    return jsonify(load_json("habit_tracker.json"))

@app.route("/api/daily_plan")
def api_daily_plan():
    return jsonify(load_json("daily_planner.json"))

@app.route("/api/weekly_report")
def api_weekly_report():
    return jsonify(load_json("weekly_report.json"))

@app.route("/api/tips")
def api_tips():
    return jsonify(load_json("tips.json"))

if __name__ == "__main__":
    app.run(debug=True)
