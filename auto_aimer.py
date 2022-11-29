from yolo import YOLO
from screen_capture import ScreenCapture
import pydirectinput as pdi
import time
import cv2 as cv
import keyboard
import threading

# config
################################################
min_confidence = 0.5
use_gpu = True
control_mouse = True
################################################

# threading function for key listener
def keyListener():
    while keyboard.read_key() != 'esc':
        pass


if __name__ == '__main__':
    # init
    yolo = YOLO(use_gpu=use_gpu)
    sc = ScreenCapture()

    # start thread for key listener to exit the program if esc pressed
    listener = threading.Thread(target=keyListener, daemon=True)
    listener.start()

    while True:
        t0 = time.time()
        img = sc.getScreenImg()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        target = yolo.findPersonTarget(img, min_confidence, display_img=False)
        if target and control_mouse:
            print(f'Move cursor to {target}')
            pdi.moveTo(target[0], target[1])
            pdi.mouseDown()
        else:
            pdi.mouseUp()

        # clear memory
        del img
        del target
        # calculate the frame rate
        fps = 1/(time.time()-t0)
        print(f'{fps:.2f} FPS')
        # exit program if escape has been pressed
        if not listener.is_alive():
            cv.destroyAllWindows()
            break