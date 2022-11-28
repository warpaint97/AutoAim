# AutoAim
 This is an auto aim bot written in python that utilizes a YOLO model from openCV to detect people on your PC's screen and takes control of your mouse to click on that person's upper body. To see how to build a YOLO model with OpenCV visit [YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)
 
### Demo Image
 ![demo](images/demo/demo.jpg "demo")
 
 ### Directory Tree
 ```
 AutoAim
│   auto_aimer.py
│   screen_capture.py
│   yolo.py
│
├───images
│       horse.jpg
│       test.jpg
│       test1.JPG
│
└───model
        coco.names
        yolov3.cfg
        yolov3.weights (missing because file size > 100 MB)
 ```
### YOLOv3 DNN weights file
The `yolov3.weights` file is 236MB large and exceeds GitHub's maximum file size of less than 100MB.
Download the pre-trained YOLO weight file [here](https://pjreddie.com/media/files/yolov3.weights).
After downloading it move it inside the `model/` directory.

### Dependencies
- (advised) **opencv-python** and **opencv-contrib-python GPU with CUDA**, see instructions [here](https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/).
- (not adviced) **opencv-python** for CPU only using `pip install opencv-python`
- **pydirectinput** for windows input synthesis
- **mss** for screen capture
- **numpy**
- **keyboard** for exit key handling
- **threading** also for the exitting
