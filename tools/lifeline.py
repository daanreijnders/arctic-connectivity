import time
import math
import argparse

def sleep(seconds=0, minutes=0):
    """
    A timer to keep connections alive. Prints amount of minutes or seconds remaining while exececuting
    Args:
        seconds: 0 by default
        minutes: 0 by default
    """
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sleep and print something every couple of minutes to keep the connection alive. Should check whether this is actually necessary.")
    parser.add_argument('minutes', type=int, help='Minutes to go to sleep')
    parser.add_argument('-s', '--seconds', default=0, type=int, help='Seconds to go to sleep.')
    args = parser.parse_args()
    sleep(seconds=args.seconds, minutes=args.minutes)
