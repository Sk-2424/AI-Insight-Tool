import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_FILE = f"{datetime.now().strftime('%d_%b_%Y')}.log"
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)


logging.basicConfig(
    # filename = LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers =[
        RotatingFileHandler(LOG_FILE_PATH, maxBytes=5000, backupCount=5)
    ]
)
if __name__ == '__main__':
    logging.info('hello world')