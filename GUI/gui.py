import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from predictors import MODELS

class PredictionGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("CSV Prediction Tool")
        self.root.geometry("420x280")

        self.csv_path = None

        # Upload button
        tk.Button(root, text="Upload CSV", width=20, command=self.upload_csv)\
            .pack(pady=10)

        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack()

        # Model selection
        tk.Label(root, text="Select Model").pack(pady=5)
        self.model_var = tk.StringVar()
        tk.OptionMenu(root, self.model_var, *MODELS.keys())\
            .pack()

        # Run
        tk.Button(root, text="Run Prediction", width=20, command=self.run)\
            .pack(pady=20)

        # Status
        self.status = tk.Label(root, text="")
        self.status.pack()

    def upload_csv(self):
        self.csv_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )
        if self.csv_path:
            self.file_label.config(text=self.csv_path)

    def run(self):
        if not self.csv_path:
            messagebox.showerror("Error", "Please upload a CSV file")
            return

        model_name = self.model_var.get()
        if model_name not in MODELS:
            messagebox.showerror("Error", "Please select a model")
            return

        try:
            self.status.config(text="Running prediction...")
            self.root.update()

            df = pd.read_csv(self.csv_path)
            predict_func = MODELS[model_name]
            df_out = predict_func(df)

            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")]
            )

            if save_path:
                df_out.to_csv(save_path, index=False)
                self.status.config(text="Prediction saved successfully ✔")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status.config(text="")

# ======================
# Run App
# ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()