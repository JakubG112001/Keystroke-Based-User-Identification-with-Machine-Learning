Keystroke Authentication System

This Python project uses keystroke dynamics and machine learning to authenticate users based on their typing patterns.

Features

- New user enrollment with typing data
- Real-time keystroke tracking
- Machine learning model using TensorFlow
- Authentication and typing pattern comparison
- Graphical visualization of typing metrics


 Step-by-Step Setup Guide

 1. Install Python if not installed

- Download Python from: https://www.python.org/downloads/
- During installation, enable the option: "Add Python to PATH"

 2. Install Visual Studio Code if not installed

- Download from: https://code.visualstudio.com/
- After installation:
  - Press Ctrl+Shift+X to open Extensions
  - Search and install the "Python" extension by Microsoft

---

Installing Dependencies

You can install all required Python libraries using one command:

```bash
pip install numpy tensorflow pynput matplotlib
or
pip install requirements.txt
```

Run this in the terminal or in the VS Code terminal.



Running the Application

From VS Code:

1. Open `KeystrokeAuthentication.py` and press Run Python File 
OR
1. Open a new terminal (`Terminal → New Terminal`)
2. Run:

```bash
python KeystrokeAuthentication.py
```

---

Enrolling a New User

1. Click "New User Enrollment"
2. Enter a username
3. Type a few lines of text normally
4. Press the ESC key to finish
5. Wait for confirmation

---

Authenticating a User

1. Click "Authenticate User"
2. Enter the same username
3. Type normally again
4. Press ESC to complete
5. View your authentication score

---

Folder Structure

```
KeystrokeAuthProject/
├── KeystrokeAuthentication.py
├── README_Guide.md
└── Python/
    ├── Users/
    └── Models/
```

These folders are created automatically on first use.

---

Troubleshooting

- Make sure Python is correctly installed
- Use Python 64-bit version
- Run VS Code as administrator if `pynput` causes issues
- Ensure the Python interpreter is selected in VS Code
- If that does not help contact author
