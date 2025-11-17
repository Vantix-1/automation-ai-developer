"""
Automation & AI Dev: Basic Calculator
Purpose: Automate arithmetic operations via CLI.
Focus: Functions, user input, and automation logic.
"""

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else "Cannot divide by zero"

# Example usage
if __name__ == "__main__":
    print("2 + 3 =", add(2, 3))
    print("10 - 4 =", subtract(10, 4))
    print("5 * 6 =", multiply(5, 6))
    print("8 / 2 =", divide(8, 2))
