import pyautogui
import time

MOVE_DISTANCE = 10   # pixels
INTERVAL = 10        # seconds between movements

while True:
    x, y = pyautogui.position()
    
    # move right
    pyautogui.moveTo(x + MOVE_DISTANCE, y, duration=0.2)
    time.sleep(1)
    
    # move back left
    pyautogui.moveTo(x, y, duration=0.2)
    
    time.sleep(INTERVAL)