import os
from datetime import datetime

class GraphLogger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = f"logs/run_{timestamp}.txt"

    def log(self, title, content):
        text = f"\n===== {title} =====\n{content}\n"
        print(text)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text)
