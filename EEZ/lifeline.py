import time
import math

def sleep(seconds=0, minutes=0):
    timer = 0
    timerMax = 60*minutes + seconds
    
    numMin = timerMax//60
    numSec = timerMax%60
    
    curMin = 0
    curSec = 0
    if numMin > 0:
        for minute in range(numMin):
            time.sleep(60)
            curMin = minute + 1
            print(f"\r Timer reached {curMin} minutes. Will stop at {numMin} minutes and {numSec} seconds.", end="")
    if numSec > 0:
        for second in range(numSec):
            time.sleep(1)
            curSec = second +1
            if numMin > 0:
                print(f"\r Timer reached {curMin} minutes and {curSec} seconds. Will stop at {numMin} minutes and {numSec} seconds.", end="")
            else:
                print(f"\r Timer reached {curSec} seconds. Will stop at {numSec} seconds.", end="")
    print("\n Timer has ended")
    return None