"""
Shows training logs
"""

class Logger:
    def __init__(self):
        self.logs = []

    def log(self, message):
        print(message)
        self.logs.append(message)
