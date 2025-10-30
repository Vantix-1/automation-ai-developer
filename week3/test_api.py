import requests
import json
import os

# ---------- SETTINGS ----------
BASE_URL = "http://127.0.0.1:5000/api"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../week2/")
TASK_FILE = os.path.join(DATA_DIR, "todo_cli_v2.json")
HABIT_FILE = os.path.join(DATA_DIR, "habit_tracker.json")

# ---------- COLOR HELPERS ----------
class c:
    OK = "\033[92m"      # Green
    WARN = "\033[93m"    # Yellow
    FAIL = "\033[91m"    # Red
    END = "\033[0m"      # Reset

# ---------- HELPERS ----------
def print_status(label, ok=True):
    color = c.OK if ok else c.FAIL
    print(f"{color}{'✅' if ok else '❌'} {label}{c.END}")

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print_status(f"Missing JSON file: {file_path}", ok=False)
        return False
    return True

def print_json_preview(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"{c.WARN}📂 {os.path.basename(file_path)} ({len(data)} records):{c.END}")
        for item in data[-3:]:  # show last 3 entries
            print("  ", item)
        print()

# ---------- API TESTS ----------
def test_api():
    print(f"{c.WARN}--- Testing Task API ---{c.END}")
    task_data = {"title": "Test Task", "priority": "High", "deadline": "2025-12-31"}

    # CREATE
    res = requests.post(f"{BASE_URL}/tasks", json=task_data)
    print_status("POST /tasks", res.status_code == 201)

    # READ
    res = requests.get(f"{BASE_URL}/tasks")
    print_status("GET /tasks", res.status_code == 200)

    # UPDATE
    if res.json():
        index = len(res.json()) - 1
        res = requests.put(f"{BASE_URL}/tasks/{index}", json={"completed": True})
        print_status("PUT /tasks/<id>", res.status_code == 200)
    else:
        print_status("PUT /tasks skipped (no data)", ok=False)

    # DELETE
    res = requests.delete(f"{BASE_URL}/tasks/0")
    print_status("DELETE /tasks/<id>", res.status_code == 200)

    print_json_preview(TASK_FILE)

    print(f"{c.WARN}--- Testing Habit API ---{c.END}")
    habit_data = {"name": "Drink Water", "frequency": "Daily"}

    # CREATE
    res = requests.post(f"{BASE_URL}/habits", json=habit_data)
    print_status("POST /habits", res.status_code == 201)

    # READ
    res = requests.get(f"{BASE_URL}/habits")
    print_status("GET /habits", res.status_code == 200)

    # UPDATE
    if res.json():
        index = len(res.json()) - 1
        res = requests.put(f"{BASE_URL}/habits/{index}", json={"completed": True})
        print_status("PUT /habits/<id>", res.status_code == 200)
    else:
        print_status("PUT /habits skipped (no data)", ok=False)

    # DELETE
    res = requests.delete(f"{BASE_URL}/habits/0")
    print_status("DELETE /habits/<id>", res.status_code == 200)

    print_json_preview(HABIT_FILE)

# ---------- RUN ----------
if __name__ == "__main__":
    print(f"{c.OK}🧠 Starting API CRUD Tests...{c.END}\n")
    if check_file_exists(TASK_FILE) and check_file_exists(HABIT_FILE):
        test_api()
    else:
        print_status("❌ Missing data files in week2/", ok=False)
