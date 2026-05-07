# Ahmed Rafallah 
# 20 - 1 - 2026

import tkinter as tk
import subprocess
import os

class App:
    def __init__(self, root):
        self.root = root
        self.root.title(" Speech Controller")
        self.root.geometry("400x200")

        self.process = None

        # Start Button (Green)
        self.start_btn = tk.Button(
            root,
            text="Start",
            width=20,
            bg="green",
            fg="white",
            command=self.start_exe
        )
        self.start_btn.pack(pady=10)

        # End Button (Red)
        self.stop_btn = tk.Button(
            root,
            text="End",
            width=20,
            bg="red",
            fg="white",
            command=self.stop_exe
        )
        self.stop_btn.pack(pady=10)

    def start_exe(self):
        if self.process is None or self.process.poll() is not None:
            exe_path = os.path.join(os.getcwd(), "vosk.exe")

            self.process = subprocess.Popen(
                exe_path,
                shell=False
            )

    def stop_exe(self):
        if self.process and self.process.poll() is None:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.process = None


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
