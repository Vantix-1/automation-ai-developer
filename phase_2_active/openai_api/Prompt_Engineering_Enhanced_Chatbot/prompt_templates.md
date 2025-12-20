# 📝 Prompt Engineering Templates
---
## 🎯 Prompt Patterns
---
### 1. Zero-shot Prompting
---
python
prompt = "Explain quantum computing in simple terms."
----
### 2. Few-shot Prompting
---
python
prompt = """
Translate English to French:

English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: I love programming.
French: J'adore la programmation.

English: The weather is beautiful today.
French:
"""
----
### 3. Chain-of-Thought
---
python
prompt = """
Let's think step by step.

Question: If a store has 15 apples and sells 3 each day, how many days until they have 6 apples left?
"""
----
### 4. Role-playing
---
python
prompt = """
You are Shakespeare. Respond in Elizabethan English.

Question: What thinkest thou of modern technology?
"""
----
##🔧 Prompt Engineering Tips
### 1. Be Specific
----
python
```
# Instead of: "Write code"
# Use: "Write a Python function that takes a list of integers and returns the sum of even numbers"
```
### 2. Provide Context
----
python
```
# Instead of: "Fix this bug"
# Use: "I'm getting a 'list index out of range' error when the input list is empty"
```

### 3. Use Examples
----
python
```
# Instead of: "Generate product names"
# Use: "Generate 5 creative product names for a coffee brand. Example: 'Morning Roast'"
```

### 4. Specify Format
----
python
```
# Instead of: "Give me book recommendations"
# Use: "Provide 3 book recommendations in JSON format"
```
### 5. Control Length
----
python
```
# Instead of: "Explain machine learning"
# Use: "Explain machine learning in 3 sentences for beginners"
```

## 📊 Temperature Guidelines
---
0.0-0.3: Factual, deterministic (coding, data analysis)

0.4-0.7: Balanced, creative but focused (general chat)

0.8-1.2: Creative, varied (storytelling, brainstorming)

1.3-2.0: Highly creative, unpredictable

## 🎮 Example Prompts
---
### Code Generation
python
```
"Write a Python function that validates email addresses using regex."
```

### Content Creation
python
```
"Write a blog post about learning Python in 2025 with 3 key benefits."
```

### Learning & Education
python
```
"Explain neural networks as if I'm 10 years old."
```

### 📝 Best Practices
----
```

Start simple, then iterate

Test multiple variations

Use delimiters for complex inputs

Specify steps for complex tasks

Limit response length when needed

Experiment with temperature

Document successful prompts

```

### 🔄 Iterative Prompt Development
----
text
```
V1: "Write about AI"
V2: "Write a 300-word article about AI ethics"
V3: "Write about AI ethics for business executives focusing on privacy"
```

### 💾 Saving and Organizing Prompts
----

#### Organize by:

Category (coding, writing, analysis)

Use case

Complexity level

Success rate

Token usage