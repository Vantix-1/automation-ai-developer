# task_api.py
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../week2/")
TASK_FILE = "todo_cli_v2.json"
HABIT_FILE = "habit_tracker.json"

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

# ----------- TASK ENDPOINTS -----------

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    tasks = load_json(TASK_FILE)
    return jsonify(tasks)

@app.route("/api/tasks", methods=["POST"])
def add_task():
    tasks = load_json(TASK_FILE)
    new_task = request.json
    new_task["completed"] = False
    tasks.append(new_task)
    save_json(TASK_FILE, tasks)
    return jsonify({"message": "Task added", "task": new_task}), 201

@app.route("/api/tasks/<int:index>", methods=["PUT"])
def update_task(index):
    tasks = load_json(TASK_FILE)
    if index < 0 or index >= len(tasks):
        return jsonify({"error": "Task not found"}), 404
    updated_task = request.json
    tasks[index].update(updated_task)
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
    return jsonify(habits)

@app.route("/api/habits", methods=["POST"])
def add_habit():
    habits = load_json(HABIT_FILE)
    new_habit = request.json
    new_habit["completed"] = False
    habits.append(new_habit)
    save_json(HABIT_FILE, habits)
    return jsonify({"message": "Habit added", "habit": new_habit}), 201

@app.route("/api/habits/<int:index>", methods=["PUT"])
def update_habit(index):
    habits = load_json(HABIT_FILE)
    if index < 0 or index >= len(habits):
        return jsonify({"error": "Habit not found"}), 404
    updated_habit = request.json
    habits[index].update(updated_habit)
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

if __name__ == "__main__":
    app.run(debug=True)
