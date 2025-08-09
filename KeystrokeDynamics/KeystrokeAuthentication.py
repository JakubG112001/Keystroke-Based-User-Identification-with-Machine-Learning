import numpy as np
from tensorflow import keras
from pynput import keyboard
from time import time, sleep
import os
import threading
import collections
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


q = collections.deque()     
dwell = []                 
startTimes = np.zeros(254)   
startTyping = 0              
DownDown = []
UpDown = []                 
virtualKeysID = []          
lastKeyEnterdTime = 0       
count = 0                   
sem = threading.Semaphore(0) 
mutex = threading.Semaphore(1) 
user_data = {}              
stop_validation = False     
stop_program = False        
similarity_check_requested = False
typing_speed = 0  
initial_data_length = 0

USERS_FOLDER = "Python/Users"
MODELS_FOLDER = "Python/Models"

if not os.path.exists(USERS_FOLDER):
    os.makedirs(USERS_FOLDER)
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

if not os.path.exists(USERS_FOLDER):
    os.makedirs(USERS_FOLDER)
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)



class TypingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Authentication System")
        self.root.geometry("600x400")
        
        self.create_widgets()
        self.main_menu()
    
    def create_widgets(self):
        self.text_display = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.text_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(self.root, text="Welcome! Please select an option from the menu.", 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.menu_frame = tk.Frame(self.root)
        self.menu_frame.pack(pady=10)
    
    def main_menu(self):
        self.clear_menu()
        
        tk.Label(self.menu_frame, text="Main Menu", font=('Arial', 14)).pack(pady=5)
        
        tk.Button(self.menu_frame, text="New User Enrollment", 
                 command=self.enroll_new_user).pack(pady=5, fill=tk.X)
        
        tk.Button(self.menu_frame, text="Authenticate User", 
                 command=self.authenticate_user).pack(pady=5, fill=tk.X)
        
        tk.Button(self.menu_frame, text="Exit", 
                 command=self.root.quit).pack(pady=5, fill=tk.X)
    
    def clear_menu(self):
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
    
    def enroll_new_user(self):
        self.clear_menu()
        self.user_name = simpledialog.askstring("New User", "Enter your username:")
        
        if self.user_name:
            if user_exists(self.user_name):
                messagebox.showinfo("Info", "User already exists. Please authenticate instead.")
                self.main_menu()
            else:
                self.text_display.delete(1.0, tk.END)
                self.status_label.config(text=f"Enrolling new user: {self.user_name}. Start typing... Press ESC when done.")
                
                threading.Thread(target=self.run_collect_data, args=(self.user_name,), daemon=True).start()
    
    def authenticate_user(self):
        self.clear_menu()
        self.user_name = simpledialog.askstring("Authenticate", "Enter your username:")
        
        if self.user_name:
            if not user_exists(self.user_name):
                messagebox.showinfo("Info", "User not found. Please enroll first.")
                self.main_menu()
            else:
                self.text_display.delete(1.0, tk.END)
                self.status_label.config(text=f"Authenticating: {self.user_name}. Start typing... Press ESC when done.")
                
                threading.Thread(target=self.run_validate_user, args=(self.user_name,), daemon=True).start()
    
    def run_collect_data(self, user_name):
        collect_data(user_name)
        self.root.after(0, self.collection_complete)
    
    def run_validate_user(self, user_name):
        global similarity_check_requested
        similarity_check_requested = False
        validate_user(user_name)
        self.root.after(0, self.validation_complete)
    
    def collection_complete(self):
        messagebox.showinfo("Success", "Data collection complete!")
        self.main_menu()
    
    def validation_complete(self):
        global similarity_check_requested, initial_data_length, dwell
        self.clear_menu()
        
 
        new_keystrokes = len(dwell) - initial_data_length if 'dwell' in globals() else 0
        
        tk.Label(self.menu_frame, text="Authentication Complete", font=('Arial', 14)).pack(pady=10)
        
        if new_keystrokes > 50:
            tk.Button(self.menu_frame, text="Improve Model with This Data", 
                     command=lambda: self.improve_model(self.user_name)).pack(pady=5, fill=tk.X)
        
        tk.Button(self.menu_frame, text="Check Typing Similarity", 
                 command=lambda: self.run_similarity_check(self.user_name)).pack(pady=5, fill=tk.X)
        
        tk.Button(self.menu_frame, text="View Typing Patterns", 
                 command=lambda: self.show_visual_results(self.user_name)).pack(pady=5, fill=tk.X)
        
        tk.Button(self.menu_frame, text="Return to Main Menu", 
                 command=self.main_menu).pack(pady=5, fill=tk.X)
    
    def improve_model(self, user_name):
        """Handle model improvement in background"""
        self.status_label.config(text="Improving model...")
        threading.Thread(target=self.run_model_improvement, args=(user_name,), daemon=True).start()
    
    def run_model_improvement(self, user_name):
        global dwell, DownDown, virtualKeysID, initial_data_length
        try:
            self.root.after(0, lambda: self.status_label.config(text="Preparing data for model improvement..."))
           
            new_dwell = dwell[initial_data_length:]
            new_DownDown = DownDown[initial_data_length:]
            new_virtualKeys = virtualKeysID[initial_data_length:]
            
            if len(new_dwell) < 30 or len(new_DownDown) < 30:
                self.root.after(0, lambda: messagebox.showerror("Error", "Not enough new data to improve the model (need ≥30 keystrokes)"))
                return
            
            self.root.after(0, lambda: self.status_label.config(text="Improving model (training epochs running)..."))
            update_model(user_name, new_dwell, new_DownDown, new_virtualKeys)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model improved with new data!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Model update failed: {str(e)}"))
        finally:
            self.root.after(0, self.main_menu)
    
    def show_visual_results(self, user_name):
        """Display graphs of typing metrics"""
        try:
            load_user_data(user_name)
            
            dwell = user_data[user_name]['dwell']
            DownDown = user_data[user_name]['DownDown']
            typing_speed = user_data[user_name].get('typing_speed', 0)
            
            result_window = tk.Toplevel(self.root)
            result_window.title(f"Typing Patterns - {user_name}")
            
            metrics_frame = tk.Frame(result_window)
            metrics_frame.pack(pady=10)
            tk.Label(metrics_frame, text=f"Typing Speed: {typing_speed:.1f} CPM", 
                    font=('Arial', 12)).pack()
            
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            ax1.plot(dwell[:100], 'b-', label='Dwell Time')
            ax1.set_title("Dwell Time (First 100 Keystrokes)")
            ax1.set_xlabel("Keystroke Index")
            ax1.set_ylabel("Time (seconds)")
            ax1.set_ylim(0, max(0.05, max(dwell[:100])*1.1))
            ax1.legend()
            
            canvas1 = FigureCanvasTkAgg(fig1, master=result_window)
            canvas1.draw()
            canvas1.get_tk_widget().pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
            
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(DownDown[:100], 'r-', label='Flight Time')
            ax2.set_title("Flight Time (First 100 Keystrokes)")
            ax2.set_xlabel("Keystroke Index")
            ax2.set_ylabel("Time (seconds)")
            ax2.legend()
            
            canvas2 = FigureCanvasTkAgg(fig2, master=result_window)
            canvas2.draw()
            canvas2.get_tk_widget().pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display results: {str(e)}")
    
    def run_similarity_check(self, user_name):
        self.text_display.delete(1.0, tk.END)
        self.status_label.config(text=f"Checking typing similarity for {user_name}. Start typing... Press ESC when done.")
        
        threading.Thread(target=self.check_similarity, args=(user_name,), daemon=True).start()
    
    def check_similarity(self, user_name):
        collect_data(user_name)
        result = check_typing_similarity(user_name)
        self.root.after(0, lambda: self.show_similarity_result(result))
    
    def show_similarity_result(self, result):
        messagebox.showinfo("Result", f"Typing similarity score: {result:.2f}")
        self.main_menu()
    
    def log_message(self, message):
        self.text_display.insert(tk.END, message + "\n")
        self.text_display.see(tk.END)
        self.root.update()

def check_typing_similarity(user_name):
    """Check similarity between current typing and stored profile"""
    try:
        load_user_data(user_name)
        user_model = keras.models.load_model(f'{MODELS_FOLDER}/{user_name}_model.h5')
        
        dwell = user_data[user_name]['dwell']
        DownDown = user_data[user_name]['DownDown']
        virtualKeysID = user_data[user_name]['virtualKeysID']
        
        if len(dwell) < 31 or len(DownDown) < 30:
            app.log_message("Insufficient typing data for comparison")
            return 0.0
        
        dwellChunk = np.array(dwell[-31:])
        DownDownChunk = np.array(DownDown[-30:])
        UpDownChunk = DownDownChunk - dwellChunk[:-1]
        
        finalVec = []
        for i in range(len(DownDownChunk)):
            inputVector = (
                virtualKeysID[i],
                virtualKeysID[i+1],
                dwellChunk[i],
                dwellChunk[i+1],
                DownDownChunk[i],
                UpDownChunk[i]
            )
            finalVec.append(inputVector)
        
        finalVec = np.array(finalVec)
        finalVec = finalVec.reshape(1, 30, 6)
        
        predictions = user_model.predict(x=finalVec, verbose=0)
        score = float(predictions[0][0])
        
        app.log_message(f"\nAuthentication Result: {score:.4f} (Threshold: 0.70)")
        return score
    except Exception as e:
        app.log_message(f"Similarity check failed: {str(e)}")
        return 0.0

def on_press(key):
    global lastKeyEnterdTime, startTyping, count
    currTime = time()

    if startTyping == 0:
        startTyping = currTime

    if lastKeyEnterdTime != 0:
        DownDown.append(currTime - lastKeyEnterdTime)
    lastKeyEnterdTime = currTime
    
    try:
        vk = key.vk if hasattr(key, 'vk') else key.value.vk
        if 0 <= vk < 254 and startTimes[vk] == 0:
            startTimes[vk] = currTime
            virtualKeysID.append(vk / 254)
            char = key.char if hasattr(key, 'char') else getattr(key.value, 'char', None)
            app.log_message(char if char else f"[{key}]")
    except Exception as e:
        app.log_message(f"Key press error: {str(e)}")

def on_release(key):
    global count, stop_program, similarity_check_requested, typing_speed
    currTime = time()
    
    try:
        vk = key.vk if hasattr(key, 'vk') else key.value.vk
        if 0 <= vk < 254:
            start = startTimes[vk]
            if start > 0:
                dwell_time = currTime - start
                if dwell_time > 0: 
                    dwell.append(dwell_time)
                    startTimes[vk] = 0 
                    
                    if count > 10 and (currTime - startTyping) > 0:
                        typing_speed = (count / (currTime - startTyping)) * 60
                    
                    if count > 30:
                        mutex.acquire()
                        q.append(count)
                        mutex.release()
                        sem.release()
                    
                    count += 1
    except Exception as e:
        app.log_message(f"Key release error: {str(e)}")
    
    if key == keyboard.Key.esc:
        stop_program = True
        return False

def prepareAndSend(count, user_model, user_name) -> float:
    global dwell, DownDown, virtualKeysID, similarity_check_requested, typing_speed

    try:
        if user_name not in user_data:
            load_user_data(user_name)

        stored_speed = user_data[user_name].get('typing_speed', 60)
        speed_similarity = min(1.5, max(0.5, typing_speed / stored_speed))

        if len(dwell) < 31 or len(DownDown) < 30:
            app.log_message("Not enough data for validation yet")
            return 0.0

        dwellChunk = np.array(dwell[count-31:count])
        DownDownChunk = np.array(DownDown[count-31:count-1])
        UpDownChunk = DownDownChunk - dwellChunk[:-1]

        finalVec = []
        index = count - 31
        for i in range(len(DownDownChunk)):
            finalVec.append([
                virtualKeysID[i+index],
                virtualKeysID[i+1+index],
                dwellChunk[i],
                dwellChunk[i+1],
                DownDownChunk[i],
                UpDownChunk[i],
                1.0 
            ])

        finalVec = np.array(finalVec).reshape(1, 30, 7)
        raw_score = float(user_model.predict(x=finalVec, verbose=0)[0][0])

       
        stored_flights = user_data[user_name]['DownDown'][-len(DownDownChunk):]
        flight_deltas = np.abs(DownDownChunk - stored_flights)
        flight_penalty = np.clip(np.mean(flight_deltas) * 5, 0, 0.3)

        adjusted_score = raw_score
        adjusted_score *= speed_similarity
        adjusted_score *= (1 - flight_penalty)
        adjusted_score = min(0.99, max(0.01, adjusted_score))

        threshold = 0.70
        if adjusted_score > threshold:
            app.log_message(f"[MATCH] Score: {adjusted_score:.4f} (Threshold: {threshold:.2f})")
            similarity_check_requested = True
        else:
            app.log_message(f"[NO MATCH] Score: {adjusted_score:.4f} (Threshold: {threshold:.2f})")

        return adjusted_score
    except Exception as e:
        app.log_message(f"Validation error: {str(e)}")
        return 0.0



def predThread(user_model, user_name):
    global stop_validation
    app.log_message("Prediction thread started")
    while not stop_validation:
        sem.acquire()
        mutex.acquire()
        if q:
            x = q.popleft()
            mutex.release()
            prepareAndSend(x, user_model, user_name)
        else:
            mutex.release()
    app.log_message("Validation stopped")

def collect_data(user_name):
    global dwell, DownDown, virtualKeysID, count, startTyping, lastKeyEnterdTime, startTimes, typing_speed
  
    dwell = []
    DownDown = []
    virtualKeysID = []
    count = 0
    startTyping = 0
    lastKeyEnterdTime = 0
    startTimes = np.zeros(254)
    typing_speed = 0

    app.log_message(f"Collecting data for {user_name}...")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

    if not DownDown or not dwell:
        app.log_message("Error: No valid keystroke data collected")
        return

    app.log_message(f"Collected {len(dwell)} dwell times (sample): {dwell[:10]}")
    app.log_message(f"Sample flight times: {DownDown[:5]}")
    app.log_message(f"Typing speed: {typing_speed:.1f} CPM")

    user_data[user_name] = {
        'dwell': np.array(dwell),
        'DownDown': np.array(DownDown),
        'virtualKeysID': np.array(virtualKeysID),
        'typing_speed': typing_speed
    }
    save_user_data(user_name)
    train_model(user_name)





def filter_outliers(data, threshold_seconds=2.0):
    """Remove outlier keystrokes with long pauses"""
    filtered = []
    for i, val in enumerate(data):
        if val < threshold_seconds:
            filtered.append(val)
        else:
            global count, initial_data_length
            count = max(0, count - 1)
            initial_data_length = max(0, initial_data_length - 1)
    return np.array(filtered)


def save_user_data(user_name):
    np.savez(f'{USERS_FOLDER}/{user_name}.npz', 
             dwell=user_data[user_name]['dwell'],
             DownDown=user_data[user_name]['DownDown'],
             virtualKeysID=user_data[user_name]['virtualKeysID'],
             typing_speed=user_data[user_name]['typing_speed'])

def load_user_data(user_name):
    global DownDown
    try:
        data = np.load(f'{USERS_FOLDER}/{user_name}.npz')
        user_data[user_name] = {
            'dwell': data['dwell'],
            'DownDown': data['DownDown'],
            'virtualKeysID': data['virtualKeysID'],
            'typing_speed': data['typing_speed'] if 'typing_speed' in data else 0
        }
        DownDown = user_data[user_name]['DownDown'].tolist()
    except Exception as e:
        app.log_message(f"Error loading user data: {str(e)}")
        DownDown = []
        raise

def train_model(user_name):
    try:
        load_user_data(user_name)
        dwell = user_data[user_name]['dwell']
        DownDown = user_data[user_name]['DownDown']
        virtualKeysID = user_data[user_name]['virtualKeysID']

        if len(dwell) < 61 or len(DownDown) < 60:
            app.log_message("Insufficient data for training (need ≥60 keystrokes)")
            return

        X, y = [], []

        for i in range(len(dwell) - 31):
            dwellChunk = dwell[i:i+31]
            DownDownChunk = DownDown[i:i+30]
            UpDownChunk = DownDownChunk - dwellChunk[:-1]

            for j in range(30):
                X.append([
                    virtualKeysID[i+j],
                    virtualKeysID[i+j+1],
                    dwellChunk[j],
                    dwellChunk[j+1],
                    DownDownChunk[j],
                    UpDownChunk[j],
                    1.0
                ])
            y.append(1)

        num_negative = len(y)
        for _ in range(num_negative):
            i = np.random.randint(0, len(dwell) - 31)
            keys = virtualKeysID[i:i+31].copy()

            dwellChunk = dwell[i:i+31] * np.random.uniform(0.85, 1.15, 31)
            DownDownChunk = DownDown[i:i+30] * np.random.uniform(0.85, 1.15, 30)

            if np.random.random() < 0.25:
                idx = np.random.randint(0, 29)
                keys[idx], keys[idx+1] = keys[idx+1], keys[idx]

            UpDownChunk = DownDownChunk - dwellChunk[:-1]

            for j in range(30):
                X.append([
                    keys[j],
                    keys[j+1],
                    dwellChunk[j],
                    dwellChunk[j+1],
                    DownDownChunk[j],
                    UpDownChunk[j],
                    1.0
                ])
            y.append(0)

        X = np.array(X).reshape(-1, 30, 7)
        y = np.array(y)

        model = keras.Sequential([
            keras.layers.LayerNormalization(input_shape=(30, 7)),
            keras.layers.LSTM(48, return_sequences=False),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X, y,
            epochs=8,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            ]
        )

        model.save(f'{MODELS_FOLDER}/{user_name}_model.h5')
        app.log_message(f"Model trained. Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    except Exception as e:
        app.log_message(f"Training failed: {str(e)}")


def update_model(user_name, new_dwell, new_DownDown, new_virtualKeys):
    try:
        load_user_data(user_name)
        user_model = keras.models.load_model(f'{MODELS_FOLDER}/{user_name}_model.h5')

        user_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0006),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        keep = 400
        combined_dwell = np.concatenate([user_data[user_name]['dwell'][-keep:], new_dwell])
        combined_DownDown = np.concatenate([user_data[user_name]['DownDown'][-keep:], new_DownDown])
        combined_virtualKeys = np.concatenate([user_data[user_name]['virtualKeysID'][-keep:], new_virtualKeys])

        X, y = [], []
        sample_weights = []

        total = len(combined_dwell) - 31
        for i in range(total):
            dwellChunk = combined_dwell[i:i+31]
            DownDownChunk = combined_DownDown[i:i+30]
            UpDownChunk = DownDownChunk - dwellChunk[:-1]
            keys = combined_virtualKeys[i:i+31]
            is_new = i >= total - len(new_dwell) - 31

            for j in range(30):
                X.append([
                    keys[j],
                    keys[j+1],
                    dwellChunk[j],
                    dwellChunk[j+1],
                    DownDownChunk[j],
                    UpDownChunk[j],
                    1.0
                ])
            y.append(1)
            sample_weights.append(1.1 if is_new else 1.0)

        for _ in range(len(y)):
            i = np.random.randint(0, len(combined_dwell) - 31)
            dwellChunk = combined_dwell[i:i+31] * np.random.uniform(0.85, 1.15, 31)
            DownDownChunk = combined_DownDown[i:i+30] * np.random.uniform(0.85, 1.15, 30)
            keys = combined_virtualKeys[i:i+31].copy()
            if np.random.random() < 0.3:
                idx = np.random.randint(0, 29)
                keys[idx], keys[idx+1] = keys[idx+1], keys[idx]

            UpDownChunk = DownDownChunk - dwellChunk[:-1]

            for j in range(30):
                X.append([
                    keys[j],
                    keys[j+1],
                    dwellChunk[j],
                    dwellChunk[j+1],
                    DownDownChunk[j],
                    UpDownChunk[j],
                    1.0
                ])
            y.append(0)
            sample_weights.append(1.0)

        X = np.array(X).reshape(-1, 30, 7)
        y = np.array(y)
        sample_weights = np.array(sample_weights)

        history = user_model.fit(
            X, y,
            sample_weight=sample_weights,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            ]
        )

        user_data[user_name] = {
            'dwell': combined_dwell,
            'DownDown': combined_DownDown,
            'virtualKeysID': combined_virtualKeys,
            'typing_speed': user_data[user_name]['typing_speed']
        }
        save_user_data(user_name)

        app.log_message(f"Model updated. Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    except Exception as e:
        app.log_message(f"Update error: {str(e)}")

    

def validate_user(user_name):
    global stop_validation, initial_data_length, dwell
    initial_data_length = len(dwell) if 'dwell' in globals() else 0
    stop_validation = False
    initial_data_length = len(dwell) 
    
    try:
        load_user_data(user_name)
        user_model = keras.models.load_model(f'{MODELS_FOLDER}/{user_name}_model.h5')
        
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        threading.Thread(target=predThread, args=(user_model, user_name), daemon=True).start()
        app.log_message(f"Authenticating {user_name}...")

        while not stop_program:
            sleep(1)
        listener.stop()
    except Exception as e:
        app.log_message(f"Validation failed: {str(e)}")

def user_exists(user_name):
    return os.path.exists(f'{USERS_FOLDER}/{user_name}.npz')


if __name__ == "__main__":
    root = tk.Tk()
    app = TypingApp(root)
    root.mainloop()