# flask_dashboard.py
from flask import Flask, render_template, request, redirect, jsonify
import requests

app = Flask(__name__)

TASKS_URL = "http://127.0.0.1:5000/api/tasks"
HABITS_URL = "http://127.0.0.1:5000/api/habits"


@app.route("/")
def home():
    try:
        tasks = requests.get(TASKS_URL).json()
        habits = requests.get(HABITS_URL).json()

        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.get("completed"))
        total_habits = len(habits)
        completed_habits = sum(1 for h in habits if h.get("completed"))

        task_completion = (
            round((completed_tasks / total_tasks) * 100, 2) if total_tasks > 0 else 0
        )
        habit_completion = (
            round((completed_habits / total_habits) * 100, 2) if total_habits > 0 else 0
        )

        return render_template(
            "dashboard.html",
            tasks=tasks,
            habits=habits,
            task_completion=task_completion,
            habit_completion=habit_completion,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- TASK CONTROLS ----------
@app.route("/add_task", methods=["POST"])
def add_task():
    name = request.form.get("task_name")
    if name:
        requests.post(TASKS_URL, json={"name": name})
    return redirect("/")


@app.route("/complete_task/<int:index>")
def complete_task(index):
    tasks = requests.get(TASKS_URL).json()
    if 0 <= index < len(tasks):
        tasks[index]["completed"] = True
        requests.put(f"{TASKS_URL}/{index}", json=tasks[index])
    return redirect("/")


@app.route("/delete_task/<int:index>")
def delete_task(index):
    requests.delete(f"{TASKS_URL}/{index}")
    return redirect("/")


# ---------- HABIT CONTROLS ----------
@app.route("/add_habit", methods=["POST"])
def add_habit():
    name = request.form.get("habit_name")
    if name:
        requests.post(HABITS_URL, json={"name": name})
    return redirect("/")


@app.route("/complete_habit/<int:index>")
def complete_habit(index):
    habits = requests.get(HABITS_URL).json()
    if 0 <= index < len(habits):
        habits[index]["completed"] = True
        requests.put(f"{HABITS_URL}/{index}", json=habits[index])
    return redirect("/")


@app.route("/delete_habit/<int:index>")
def delete_habit(index):
    requests.delete(f"{HABITS_URL}/{index}")
    return redirect("/")


if __name__ == "__main__":
    app.run(port=5001, debug=True)
