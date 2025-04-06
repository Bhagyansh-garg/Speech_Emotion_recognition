import customtkinter as ctk  # Modern UI
from tkinter import ttk, Frame, StringVar  # For Tabs
import threading  # For Smooth UI Updates
import matplotlib.pyplot as plt
from PIL import Image, ImageTk  # For displaying images
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from audio import upload_audio, record_audio
import os

# Emoji mapping for emotions with colors
EMOTION_COLORS = {
    "neutral": ("😐 Neutral", "#808080"),  # Gray
    "calm": ("😌 Calm", "#87CEEB"),  # Light Blue
    "happy": ("😊 Happy", "#32CD32"),  # Green
    "sad": ("😢 Sad", "#1E90FF"),  # Blue
    "angry": ("😡 Angry", "#FF4500"),  # Red-Orange
    "fearful": ("😨 Fearful", "#8A2BE2"),  # Purple
    "disgust": ("🤢 Disgust", "#8B4513"),  # Brown
    "surprised": ("😲 Surprised", "#FFD700"),  # Gold
}

recent_predictions = []  # Store recent emotions for visualization

# Set up modern theme (Light Mode)
ctk.set_appearance_mode("light")  
ctk.set_default_color_theme("blue")

# Initialize root window
root = ctk.CTk()
root.title("🎙 Speech Emotion Recognition")
root.geometry("750x700")
root.minsize(750, 600)  # Prevent accidental shrinking
root.configure(bg="#E3F2FD")  # Light blue background

style = ttk.Style()
style.configure("TNotebook.Tab", font=("Arial", 14, "bold"))  # Increase tab text size

# Create notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

# Function to create styled frames with different colors
def create_colored_frame(parent, bg_color):
    frame = ctk.CTkFrame(parent, fg_color=bg_color)
    return frame

# Create frames for each tab
record_tab = create_colored_frame(notebook, "#FFF3E0")  # Light orange
upload_tab = create_colored_frame(notebook, "#E8F5E9")  # Light green
chart_tab = create_colored_frame(notebook, "#E3F2FD")  # Light blue

notebook.add(record_tab, text=" 🎤 Record Audio ")
notebook.add(upload_tab, text=" 📂 Upload Audio ")
notebook.add(chart_tab, text=" 📊 Class Distribution ")

# Variable to store the selected chart type
chart_type = StringVar(value="bar")  # Default to bar chart

# Function to display the selected chart
def show_chart():
    selected_chart = "bar_chart.png" if chart_type.get() == "bar" else "pie_chart.png"

    if os.path.exists(selected_chart):  # Check if file exists
        img = Image.open(selected_chart)
        img = img.resize((500, 350), Image.LANCZOS)  # Resize to fit the tab
        img = ImageTk.PhotoImage(img)

        chart_label.configure(image=img)
        chart_label.image = img  # Keep a reference to prevent garbage collection
    else:
        chart_label.configure(image="", text="Chart not found!", font=("Arial", 16, "bold"), text_color="black")

# Function to update emotion display
def display_emotion(emotion, label):
    emoji_text, color = EMOTION_COLORS.get(emotion, ("❓ Unknown", "#000000"))  # Default black if unknown
    label.configure(text=emoji_text, text_color=color)

    # Update recent predictions list
    recent_predictions.append(emotion)
    if len(recent_predictions) > 5:
        recent_predictions.pop(0)

    update_recent_predictions()
    update_graph()

# Function to update recent emotions
def update_recent_predictions():
    recent_label.configure(text="Last 5 Emotions:\n" + "  |  ".join(recent_predictions), text_color="black")

# Function to handle recording with UI updates
def start_recording():
    record_status.configure(text="🎤 Recording...", text_color="#D84315")  # Dark orange
    progress_bar.start(10)  # Start animation
    root.update_idletasks()

    # Run recording in a separate thread
    def record():
        emotion = record_audio()
        progress_bar.stop()  # Stop animation
        record_status.configure(text="✅ Recording Finished!", text_color="#2E7D32")  # Dark green
        display_emotion(emotion, emotion_label_record)

    threading.Thread(target=record).start()

# Function to update graph
def update_graph():
    if not recent_predictions:
        return  # Avoid updating if no data

    # Count occurrences of each emotion
    emotion_counts = {emotion: recent_predictions.count(emotion) for emotion in set(recent_predictions)}

    # Clear old graph
    ax.clear()
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color=[EMOTION_COLORS[e][1] for e in emotion_counts])
    ax.set_ylabel("Frequency")
    ax.set_title("Emotion Analysis")

    canvas.draw()

# --------------- RECORD TAB UI ---------------
title_label_record = ctk.CTkLabel(record_tab, text="🎙 Record & Detect Emotion", font=("Arial", 22, "bold"), text_color="#BF360C")  
title_label_record.pack(pady=15)

emotion_label_record = ctk.CTkLabel(record_tab, text="🎭 Detected Emotion: ?", font=("Arial", 20, "bold"), fg_color="white", width=250, text_color="black")
emotion_label_record.pack(pady=10)

progress_bar = ttk.Progressbar(record_tab, mode="indeterminate", length=300)
progress_bar.pack(pady=10)

record_status = ctk.CTkLabel(record_tab, text="", font=("Arial", 14, "bold"), text_color="#BF360C")
record_status.pack(pady=5)

btn_record = ctk.CTkButton(record_tab, text="🎤 Start Recording", fg_color="#FF4500", hover_color="#CC3700", text_color="white", command=start_recording)
btn_record.pack(pady=10)

# --------------- UPLOAD TAB UI ---------------
title_label_upload = ctk.CTkLabel(upload_tab, text="📂 Upload Audio & Detect Emotion", font=("Arial", 22, "bold"), text_color="#1B5E20")
title_label_upload.pack(pady=15)

emotion_label_upload = ctk.CTkLabel(upload_tab, text="🎭 Detected Emotion: ?", font=("Arial", 20, "bold"), fg_color="white", width=250, text_color="black")
emotion_label_upload.pack(pady=10)

btn_upload = ctk.CTkButton(upload_tab, text="📂 Select & Upload File", fg_color="#008080", hover_color="#006666", text_color="white",
                           command=lambda: display_emotion(upload_audio(), emotion_label_upload))
btn_upload.pack(pady=10)

# --------------- RECENT PREDICTIONS PANEL ---------------
recent_label = ctk.CTkLabel(root, text="Last 5 Emotions: ", font=("Arial", 16, "bold"), text_color="black")
recent_label.pack(pady=10)

# --------------- GRAPH VISUALIZATION ---------------
graph_frame = Frame(root)
graph_frame.pack(pady=10, expand=True)

fig, ax = plt.subplots(figsize=(5, 2))
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack()

# --------------- CHART TAB UI ---------------
title_label_chart = ctk.CTkLabel(chart_tab, text="📊 Emotion Class Distribution", 
                                 font=("Arial", 22, "bold"), text_color="#0D47A1")
title_label_chart.pack(pady=10)

# Radio buttons to select chart type
radio_frame = ctk.CTkFrame(chart_tab, fg_color="#E3F2FD")
radio_frame.pack(pady=5)

radio_bar = ctk.CTkRadioButton(radio_frame, text="📊 Bar Chart", variable=chart_type, value="bar", command=show_chart)
radio_pie = ctk.CTkRadioButton(radio_frame, text="🥧 Pie Chart", variable=chart_type, value="pie", command=show_chart)

radio_bar.grid(row=0, column=0, padx=10)
radio_pie.grid(row=0, column=1, padx=10)

# Label to display the chart
chart_label = ctk.CTkLabel(chart_tab, text="")  # Initially empty
chart_label.pack(pady=10)

# Ensure the correct chart loads when the tab is opened
show_chart()
# --------------- EXIT BUTTON ---------------
btn_exit = ctk.CTkButton(root, text="❌ Exit", fg_color="#D32F2F", hover_color="#B71C1C", text_color="white",
                         font=("Arial", 14, "bold"), corner_radius=10, command=root.quit)
btn_exit.pack(side="bottom", pady=15)

# Run Tkinter event loop
root.mainloop()
