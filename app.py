import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os

# 1. Settings / Configuration
MODEL_PATH = 'emnist_model.h5'  # Path to the trained model file
CANVAS_SIZE = 280               # Drawing canvas size (large for user)
INPUT_SIZE = 28                 # Input size for the Neural Network (28x28)
PEN_WIDTH = 15                  # Brush stroke width (must be thick like dataset images)

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Handwriting Recognition")
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}\nEnsure {MODEL_PATH} is in the same directory.")
            root.destroy()
            return

        # Label mapping (0-46 -> Characters) based on EMNIST Balanced dataset
        self.mapping = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
        }

        # UI Setup
        # We draw white on black because the model was trained on inverted images
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor="cross")
        self.canvas.pack(pady=10)

        # Create a "hidden" PIL image to draw on in memory
        # Mode 'L' means 8-bit pixels, black and white
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0) 
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint) # Left mouse button drag

        # Buttons Frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.btn_predict = tk.Button(btn_frame, text="Recognize", command=self.predict_digit, font=("Helvetica", 14), bg="#4CAF50", fg="white")
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, font=("Helvetica", 14), bg="#f44336", fg="white")
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        # Result Label
        self.label_result = tk.Label(root, text="Draw a digit or character...", font=("Helvetica", 16))
        self.label_result.pack(pady=10)

    def paint(self, event):
        # Draw on the GUI Canvas
        x1, y1 = (event.x - PEN_WIDTH), (event.y - PEN_WIDTH)
        x2, y2 = (event.x + PEN_WIDTH), (event.y + PEN_WIDTH)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # Draw on the PIL Image (in memory)
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        # Reset the memory image
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Canvas cleared")

    def predict_digit(self):
        # 1. Resize image from 280x280 to 28x28 (Model input size)
        img_resized = self.image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
        
        # 2. Convert to NumPy array
        img_array = np.array(img_resized)
        
        # 3. Preprocessing for the model:
        # Normalize (0-255 -> 0.0-1.0) and reshape to (Batch, Height, Width, Channels)
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)

        # 4. Prediction
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)      # Class with highest probability
        probability = np.max(prediction)             # Confidence score

        # 5. Display Result
        char = self.mapping.get(predicted_class, "?")
        self.label_result.config(text=f"Prediction: {char} ({probability*100:.1f}%)")
        print(f"Predicted: {char}, Confidence: {probability:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()