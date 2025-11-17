from flask import Flask, jsonify, request
import json, os, uuid

app = Flask(__name__)

DATA_FILE = "data/tasks.json"

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Initialize file if missing
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)


def load_tasks():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_tasks(tasks):
    with open(DATA_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


# -----------------------
#       API ROUTES
# -----------------------

@app.get("/tasks")
def get_tasks():
    return jsonify(load_tasks())


@app.post("/tasks")
def add_task():
    data = request.json
    tasks = load_tasks()

    new_task = {
        "id": str(uuid.uuid4()),
        "title": data.get("title", "Untitled Task"),
        "category": data.get("category", "general"),
        "priority": data.get("priority", "medium"),
        "completed": False
    }

    tasks.append(new_task)
    save_tasks(tasks)

    return jsonify(new_task), 201


@app.put("/tasks/<task_id>")
def update_task(task_id):
    tasks = load_tasks()

    for t in tasks:
        if t["id"] == task_id:
            t.update(request.json)
            save_tasks(tasks)
            return jsonify(t)

    return jsonify({"error": "Task not found"}), 404


@app.patch("/tasks/<task_id>/toggle")
def toggle_task(task_id):
    tasks = load_tasks()

    for t in tasks:
        if t["id"] == task_id:
            t["completed"] = not t["completed"]
            save_tasks(tasks)
            return jsonify(t)

    return jsonify({"error": "Task not found"}), 404


@app.delete("/tasks/<task_id>")
def delete_task(task_id):
    tasks = load_tasks()
    new_tasks = [t for t in tasks if t["id"] != task_id]

    save_tasks(new_tasks)

    return jsonify({"message": "Task deleted"}), 200


if __name__ == "__main__":
    app.run(port=5001, debug=True)
