# task_api.py
from flask import Flask, request, jsonify, render_template
import json
import os

app = Flask(__name__, template_folder="templates")

DATA_DIR = os.path.join(os.path.dirname(__file__), "week2")
TASK_FILE = "todo_cli_v2.json"
HABIT_FILE = "habit_tracker.json"

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def load_json(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_json(file_name, data):
    path = os.path.join(DATA_DIR, file_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def initialize_files():
    # Preload tasks if empty
    if not load_json(TASK_FILE):
        sample_tasks = [
            {"name": "Finish report", "completed": False, "priority": "High", "deadline": "2025-11-05"},
            {"name": "Check emails", "completed": False, "priority": "Medium", "deadline": "2025-10-30"}
        ]
        save_json(TASK_FILE, sample_tasks)

    # Preload habits if empty
    if not load_json(HABIT_FILE):
        sample_habits = [
            {"name": "Meditate", "completed": False, "streak": 0, "last_completed": None},
            {"name": "Exercise", "completed": False, "streak": 0, "last_completed": None}
        ]
        save_json(HABIT_FILE, sample_habits)

# Initialize files with dummy data if empty
initialize_files()

# ----------- TASK ENDPOINTS -----------
@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    tasks = load_json(TASK_FILE)
    # Sort by priority (High → Medium → Low) then by nearest deadline
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    tasks.sort(key=lambda t: (
        priority_order.get(t.get("priority", "Medium"), 1),
        t.get("deadline") or ""
    ))
    return jsonify(tasks)

@app.route("/api/tasks", methods=["POST"])
def add_task():
    tasks = load_json(TASK_FILE)
    new_task = request.json
    new_task.setdefault("completed", False)
    tasks.append(new_task)
    save_json(TASK_FILE, tasks)
    return jsonify({"message": "Task added", "task": new_task}), 201

@app.route("/api/tasks/<int:index>", methods=["PUT"])
def update_task(index):
    tasks = load_json(TASK_FILE)
    if index < 0 or index >= len(tasks):
        return jsonify({"error": "Task not found"}), 404
    tasks[index].update(request.json)
    save_json(TASK_FILE, tasks)
    return jsonify({"message": "Task updated", "task": tasks[index]})

@app.route("/api/tasks/<int:index>", methods=["DELETE"])
def delete_task(index):
    tasks = load_json(TASK_FILE)
    if index < 0 or index >= len(tasks):
        return jsonify({"error": "Task not found"}), 404
    removed_task = tasks.pop(index)
    save_json(TASK_FILE, tasks)
    return jsonify({"message": "Task deleted", "task": removed_task})

# ----------- HABIT ENDPOINTS -----------
@app.route("/api/habits", methods=["GET"])
def get_habits():
    habits = load_json(HABIT_FILE)
    # Sort by streak descending
    habits.sort(key=lambda h: h.get("streak", 0), reverse=True)
    return jsonify(habits)

@app.route("/api/habits", methods=["POST"])
def add_habit():
    habits = load_json(HABIT_FILE)
    new_habit = request.json
    new_habit.setdefault("completed", False)
    new_habit.setdefault("streak", 0)
    new_habit.setdefault("last_completed", None)
    habits.append(new_habit)
    save_json(HABIT_FILE, habits)
    return jsonify({"message": "Habit added", "habit": new_habit}), 201

@app.route("/api/habits/<int:index>", methods=["PUT"])
def update_habit(index):
    habits = load_json(HABIT_FILE)
    if index < 0 or index >= len(habits):
        return jsonify({"error": "Habit not found"}), 404
    habits[index].update(request.json)
    save_json(HABIT_FILE, habits)
    return jsonify({"message": "Habit updated", "habit": habits[index]})

@app.route("/api/habits/<int:index>", methods=["DELETE"])
def delete_habit(index):
    habits = load_json(HABIT_FILE)
    if index < 0 or index >= len(habits):
        return jsonify({"error": "Habit not found"}), 404
    removed_habit = habits.pop(index)
    save_json(HABIT_FILE, habits)
    return jsonify({"message": "Habit deleted", "habit": removed_habit})

# ----------- DASHBOARD ROUTE -----------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")  # Flask will load it from week3/templates


if __name__ == "__main__":
    app.run(debug=True)