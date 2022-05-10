# Security Camera Monitor
This is a simple rust project using opencv, Coral EdgeTPU and TensorflowLite to detect people in a video stream. 

Features :-
* Run with UI monitor window or headless,
* Open RTSP stream (most cheap IP cameras have this),
* Detects people,
* Define a boundary polygon to trigger recording,
* Records to video file and grabs first and best photo image,
* Configure notifications using a shell script, e.g. send to Telegram or upload to cloud storage like S3,
* Optional continuous timelapse recording at 1 fps,

This project uses the ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite model as this was a pretty good balance between accuracy and performance.
It supports all the COCO image labels so if you wanted to detect say dogs or birds instead of people then you could easily change this in the source code. 

You'll need the following dependencies installed :-
* libedgetpu (source: https://github.com/google-coral/libedgetpu)
* libtensorflowlite-c this is the C api to tensorflowlite. This has to be built from the same revision expected by libedgetpu (see above link). 
* opencv 

You could also use the non-edgetpu ssdlite_mobiledet_coco model and then comment out the edgetpu code and that should work with any tensorflowlite. This could be CPU based (slow) or GPU based on say the Jetson Nano. 
The cheapest option is probably the Coral USB accelerator and a Raspberry Pi. 

## Usage
```text
security_camera
Person activated camera video stream monitoring and recording

Usage:
  security_camera [options] <video-source>
  security_camera [options]
  security_camera (-h | --help)

Options:
  -h --help                         Show this screen
  -m --monitor                      Create monitor window showing real time feed
  -t --timelapse                    Record timelapse files, continuous 1 fps with hourly rollover
  -p --polygon <polygon-file>       Use a boundary polygon, polygon file is csv with one point per line
```

## Person Detection and Boundary Polygon
Using the deep learning model, each person detected will be enclosed in a bounding rectangle. The software, calculates the centre point of the rectangle and checks to see whether that point falls within the boundary polygon. If it does then it starts recording.
It keeps a buffer of 150 frames (around 10 seconds) in memory constantly so when it starts the recording you'll have the 10 seconds leading up to the first detection. It then continues recording until 30 seconds have passed without any further detection. 
This should give complete coverage of the person's activity. 

Since your camera may cover not just your area but possibly a neighbour's or a public street, you can define a boundary polygon so that you don't get false alarms everytime somebody walks down the street. A polygon is an enclosed shape consisting of a number of points with the last one being the same as the first and therefore closing the shape.
If no polygon is defined, then the detection area is the entire centre square of the frame (see Video Considerations below).

Define a polygon in csv file e.g. 'polygon.csv' like this :-
```text
x,y
0,100
160,160
280,130
280,70
380,50
720,50
720,720
0,720
0,100
```
Each line is one point in x,y format at the original frame resolution but for the centre square (see Video Considerations below). The polygon will be drawn in monitor mode so you can see if it's correct or not. 
Sorry there's no GUI, you'll just have to use trial and error. I guess you could take a screen shot and use some image software to read off the points.

## Timelapse Recording
Timelapse recording will create a one video file for each hour containing 1 frame per second so you have a continual record from the camera.
Typically these files will be quite large (600mb per hour for a HD stream) and so you'll probably want to use the notify_timelapse_rollover.sh to implement some data management.

## Notifications
There are 3 'events' which can be used to trigger notifications via shell scripts with the same name :-
 1. notify_start_person.sh will be called when a person is first detected within the boundary polygon. The first argument passed is the image filename of the first frame.
 2. notify_end_person.sh will be called when the video is complete. The first argument is the 'best' image (largest rectangle) captured and the 2nd argument is the video filename.
 3. notify_timelapse_rollover.sh will be called as timelapse video is closed and new one created. The first argument is the filename of the just closed video filename.

Using these, you can create responses to particular events. For example I send a Telegram message using telegram-cli and also upload the videos to AWS S3.
When security_camera starts up, it checks for the existence of these files in the current directory and then will use them if they exist. If any don't exist then there will be no notifications of that type.


## Video Stream Considerations
Most cameras are likely to be 'HD' i.e. 1080p in a wide aspect ratio. The input tensor is 320x320x3(RGB) which is obviously square. My approach is to take the large square from the centre of the frame and lose the left and right edges.
Monitor mode shows this but recordings will be the full frame width. This means that if your 'region of interest' (ROI) is at the far left or right of the frame then this isn't going to work too well for you. The centre square is then resized to 320x320, colors remapped from BGR (opencv's format) to RGB and then passed to the model.
The output tensor contains the objects detected (1 = person) and bounding rectangles which are scaled back up to the original resolution.

If you don't like this approach, there are some alternatives you could try :- 
 1. Scale the entire frame to 320x320 with a blank area top or bottom. This will lose resolution and may affect acurracy,
 2. Create 2 squares left and right with an overlap in the middle and call the model twice. This will obviously halve the performance,
 3. Simply shift the square left or right depending on your ROI (maybe this could be tied into the boundary polygon to make it a bit smarter)

## Licence and Commercial Use
This code is provided as open source under the GPL3 licence primarily because there aren't a lot of examples of using machine learning with Rust. 
Please feel free to use and modify this for your personal use. If you want to use it commercially then please email me at andrew [at] trailgauge [dot] com.
I'm happy to work collaboratively on a commercial application or license as is very reasonably.