import os
from datetime import datetime

def log(text, file_name=None):
    # Ensure the 'logs' directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Generate log file name based on today's date
    today = datetime.now().strftime('%Y-%m-%d')
    log_file_path = os.path.join(logs_dir, f'{today}.log')

    # Append log entry to the log file
    with open(log_file_path, 'a') as log_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{timestamp}] {file_name}: {text}\n')

