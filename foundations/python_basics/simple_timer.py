"""
Automation & AI Dev: Focus Timer
Purpose: CLI-based focus timer to automate work sessions.
Focus: Loops, time module, and user notifications.
"""

import time

def countdown(minutes):
    print(f"Starting {minutes}-minute focus timer...")
    try:
        for remaining in range(minutes * 60, 0, -1):
            mins, secs = divmod(remaining, 60)
            print(f"{mins:02d}:{secs:02d}", end="\r")
            time.sleep(1)
        print("\nTime's up! Take a break or continue working.")
    except KeyboardInterrupt:
        print("\nTimer stopped manually.")

if __name__ == "__main__":
    countdown(1)  # Example: 1-minute timer for testing
