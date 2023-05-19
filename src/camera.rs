use std::time::SystemTime;
use std::process::Command;
use std::thread;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::{Arc, mpsc, Mutex};
use chrono::{DateTime, Local, Timelike};

use serde::Deserialize;
use opencv::{Error, highgui, prelude::*, Result, videoio};

use moonfire_tflite::*;
use opencv::core::{Rect, Scalar, Size, Vector};
use opencv::imgproc::{INTER_AREA, line, rectangle, resize};
use opencv::imgcodecs::imwrite;
use opencv::videoio::{VideoCapture, VideoWriter};

const LINE_8: i32 = 8;

const RESOLUTION: i32 = 320;  // input tensor resolution
const THRESHOLD: f32 = 0.75;
const MAX_BUFFER_FRAMES: usize = 15 * 120;

#[derive(Debug, Deserialize, Clone)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point
{
    pub fn new(x: i32, y: i32) -> Point
    {
        Point {x, y }
    }
}

type Polygon = Vec<Point>;

#[derive(Debug, Deserialize, Clone)]
pub struct Camera {
    pub name: String,
    pub source: String,
    pub timelapse: bool,
    pub monitor: bool,
    pub boundary: Option<Polygon>,
    pub trigger_frames: i32,
    pub trigger_distance: f32,
}

#[derive(Clone)]
enum FrameSend {
    Frame(Mat),
    Best(Mat, String),
    End,
}

impl Camera
{
    pub fn new(url: &str) -> Camera {
        Camera {
            name: "Security Camera".to_string(),
            source: url.to_string(),
            timelapse: false,
            monitor: true,
            boundary: None,
            trigger_frames: 1,
            trigger_distance: 0.0,
        }
    }

    pub fn run(&self, interpreter: Arc<Mutex<Interpreter>>, notify_start_person: bool, notify_end_person: bool, notify_timelapse_rollover: bool) -> Result<()>
    {
        let mut shutdown = false;
        if self.monitor
        {
            highgui::named_window(&self.name, highgui::WINDOW_AUTOSIZE)?;
            info!("Opened monitor window for {}", &self.name);
        }


        let mut cam: VideoCapture;

        /* Redundant
        #[cfg(ocvrs_opencv_branch_32)]
        if self.source.len() == 0
        {
            cam = videoio::VideoCapture::new_default(0)?;
            info!("{}: Opening default video stream.", &self.name);
        } else {
            cam = videoio::VideoCapture::open(&self.source);
            info!("{}: Opening video stream at {}", &self.name, &self.source);
        }

        #[cfg(not(ocvrs_opencv_branch_32))]

        if self.source.len() == 0
        {
            cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
            // 0 is the default camera
            info!("{}: Opening default video stream.", &self.name);
        } else { */

        cam = videoio::VideoCapture::from_file(&self.source, 0)?;

        // stream
        info!("{}: Opening video stream at {}", &self.name, &self.source);
        //}

        let opened = videoio::VideoCapture::is_opened(&cam)?;

        if !opened {
            error!("{}: Can't open video stream !", &self.name);
            return Err(Error::new(-1, "Camera aborted"));
        }

        // Initialisation
        let mut frame = Mat::default();
        let mut frame320 = Mat::default();
        let size320 = Size::new(RESOLUTION, RESOLUTION);
        let mut fx: i32 = 0;
        let mut fw = 0;
        let mut d = 0.0;
        let mut fsize = Size::new(0, 0);

        // Get video stream size - input tensor is 320 x 320 RGB so we'll take the central square from the video stream
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            let w = frame.size()?.width;
            let h = frame.size()?.height;
            d = (w as f32) / (RESOLUTION as f32);
            let dy = (h as f32) / (RESOLUTION as f32);
            if dy < d { d = dy; }
            fx = (w - (((RESOLUTION as f32) * d) as i32)) / 2;
            fw = fx + (((RESOLUTION as f32) * d) as i32);
            fsize = frame.size()?.clone();
        }

        let mut tick = SystemTime::now();
        let mut frames = 0;
        let mut fps = 0.0;
        let mut frames_minute = 0;
        let mut elapsed_seconds = 0;
        let mut last_minute_fps = 0;


        // Timelapse recording
        let mut timelapse_filename = "".to_string();
        let mut timelapse: VideoWriter = VideoWriter::default()?;
        let mut skip_timlapse = false;
        if self.timelapse
        {
            info!("{}: Timelapse recording is enabled.", &self.name);
            timelapse_filename = format!("captures/timelapse/{}{}.mp4", &self.name, timestamp_string());
            timelapse = create_video_writer(&timelapse_filename, 1.5, fsize);
        } else { info!("{}: Timelapse recording is disabled.", &self.name); }

        // Person recording
        let buffer_size = 150;
        let mut buffer_pnt = 0;
        let mut buffer: Vec<Mat> = Vec::with_capacity(buffer_size);    /* Cyclic buffer for 10 seconds prior to detection */
        let mut person_recording = false;
        let mut person_best_size = 0;
        let mut person_last_seen = SystemTime::now();
        let mut person_trigger_frames_person = 0;
        let mut person_trigger_distance = 0.0;
        let mut person_trigger_last_x = 0;
        let mut person_trigger_last_y = 0;


        // Channel to send frames
        let mut sync_sender: Option<Sender<FrameSend>> = None;

        // Main activity loop
        loop {
            let rs = cam.read(&mut frame);
            match rs {
                Ok(true) => {
                    if frame.size()?.width > 0 {
                        let frame320rc = frame.col_bounds(fx, fw);
                        match frame320rc {
                            Ok(mut frame320rc) => {
                                resize(&frame320rc, &mut frame320, size320, 0.0, 0.0, INTER_AREA);

                                // Call the interpreter
                                let person = person_in_frame(&interpreter, &frame320, d);
                                if let Some(r) = person
                                {
                                    let outside_color = Scalar::from((64.0, 64.0, 240.0));
                                    let inside_color = Scalar::from((64.0, 240.0, 64.0));

                                    let centre = Point::new(r.x + r.width / 2, r.y + r.height / 2);

                                    if inside_polygon(&self.boundary, &centre)
                                    {
                                        if self.monitor
                                        {
                                            rectangle(&mut frame320rc, r, inside_color, 2, LINE_8, 0);
                                        }

                                        person_last_seen = SystemTime::now();
                                        let area = r.height * r.width;
                                        if area > person_best_size
                                        {
                                            person_best_size = area;
                                            let person_best_frame = frame.clone();
                                            let person_best_time = timestamp_string();

                                            match &sync_sender
                                            {
                                                Some(tx) => { tx.send(FrameSend::Best(person_best_frame, person_best_time)); }
                                                None => {}
                                            }
                                        }

                                        person_trigger_frames_person += 1;
                                        if (person_trigger_last_x == 0) && (person_trigger_last_y == 0)
                                        {
                                            person_trigger_last_x = centre.x;
                                            person_trigger_last_y = centre.y;
                                        }
                                        let dx = (centre.x - person_trigger_last_x) as f32;
                                        let dy = (centre.y - person_trigger_last_y) as f32;
                                        person_trigger_distance += f32::sqrt(dx * dx + dy * dy);

                                        if !person_recording && (person_trigger_frames_person > self.trigger_frames) && (person_trigger_distance > self.trigger_distance)
                                        {
                                            // Start recording
                                            info!("Person detected - recording started to buffer");
                                            person_recording = true;

                                            // start the async writer
                                            let (tx, rx) = mpsc::channel();

                                            let video_filename = format!("captures/people/video/{}{}.mp4", self.name, timestamp_string());
                                            let image_filename = format!("captures/people/photos/{}{}-first.jpg", self.name, timestamp_string());
                                            async_writer(rx, video_filename, image_filename.clone(), fps, fsize, notify_end_person, self.name.clone());

                                            // Write the cyclic buffer frames
                                            for _ in buffer_pnt..(buffer.len() - 1)
                                            {
                                                let f = buffer.remove(buffer_pnt);
                                                tx.send(FrameSend::Frame(f));
                                            }
                                            if buffer_pnt > 0 {
                                                for _ in 0..(buffer_pnt - 1)
                                                {
                                                    let f = buffer.remove(0);
                                                    tx.send(FrameSend::Frame(f));
                                                }
                                            }
                                            buffer_pnt = 0;
                                            sync_sender = Some(tx);

                                            // Write first photo and call notifier
                                            let flags = Vector::new();
                                            imwrite(&image_filename, &frame, &flags);
                                            if notify_start_person
                                            {
                                                info!("Calling 'notify_start_person.sh {}'", &image_filename);
                                                let r = Command::new("./notify_start_person.sh")
                                                    .arg(&image_filename).spawn();
                                                if let Err(e) = r { error!("Error calling script: {}", e) }
                                            }
                                        }
                                    } else {
                                        if self.monitor
                                        {
                                            rectangle(&mut frame320rc, r, outside_color, 2, LINE_8, 0);
                                        }
                                    }
                                }


                                // Person recording
                                if person_recording
                                {
                                    let elapsed = SystemTime::now().duration_since(person_last_seen).unwrap().as_millis();

                                    match &sync_sender
                                    {
                                        Some(tx) => {
                                            if elapsed > 30000 { tx.send(FrameSend::End); } else { tx.send(FrameSend::Frame(frame.clone())); }
                                        }
                                        None => { error!("sync_sender is none."); }
                                    }

                                    if elapsed > 30000  // 30 seconds since last activity
                                    {
                                        // Finish the async writing
                                        person_recording = false;
                                        person_best_size = 0;
                                        buffer_pnt = 0;
                                    }
                                } else {
                                    // Stash the frame in the buffer
                                    if buffer.len() <= buffer_pnt { buffer.push(frame.clone()); } else { buffer[buffer_pnt] = frame.clone(); }
                                    buffer_pnt = buffer_pnt + 1;
                                    if buffer_pnt == buffer_size { buffer_pnt = 0 };
                                }

                                // Time related
                                frames += 1;
                                frames_minute += 1;

                                let elapsed = SystemTime::now().duration_since(tick).unwrap().as_millis();
                                if elapsed > 1000
                                {
                                    // Update fps
                                    fps = (frames as f64) / ((elapsed as f64) / 1000.0);
                                    frames = 0;
                                    elapsed_seconds += 1;
                                    if elapsed_seconds >= 300 // 5 minutes
                                    {
                                        let fps = frames_minute / elapsed_seconds;
                                        if fps != last_minute_fps
                                        {
                                            info!("{}: Average fps = {:.1}", &self.name, frames_minute as f32 / elapsed_seconds as f32);
                                            last_minute_fps = fps;
                                        }
                                        elapsed_seconds = 0;
                                        frames_minute = 0;
                                    }


                                    if !person_recording && (person_trigger_frames_person > 0) { info!("Failed trigger, frames: {:}, distance: {:}", person_trigger_frames_person, person_trigger_distance); }
                                    person_trigger_frames_person = 0;
                                    person_trigger_distance = 0.0;
                                    person_trigger_last_x = 0;
                                    person_trigger_last_y = 0;

                                    tick = SystemTime::now();

                                    if self.timelapse
                                    {
                                        // Write timelapse frame
                                        timelapse.write(&frame)?;

                                        // Rollover timelapse file
                                        let time = Local::now();
                                        if (time.second() == 0) && (time.minute() == 0) && (!skip_timlapse)
                                        {
                                            skip_timlapse = true;
                                            timelapse.release()?;

                                            if notify_timelapse_rollover
                                            {
                                                // Call the notify script
                                                info!("Calling 'notify_timelapse_rollover.sh {}'", &timelapse_filename);
                                                let r = Command::new("./notify_timelapse_rollover.sh")
                                                    .arg(&timelapse_filename).spawn();
                                                if let Err(e) = r { error!("Error calling script: {}", e) }
                                            }

                                            timelapse_filename = format!("captures/timelapse/{}{}.mp4", &self.name, timestamp_string());
                                            timelapse = create_video_writer(&timelapse_filename, 1.5, fsize);
                                        } else { skip_timlapse = false; }
                                    }
                                }

                                if self.monitor
                                {
                                    if let Some(polygon) = &self.boundary { draw_boundary(polygon, &mut frame320rc); }
                                    highgui::imshow(&self.name, &mut frame)?;
                                }
                            }
                            Err(e) => { error!("Error extracting columns from frame: {}", e); }
                        }
                    }
                    if self.monitor
                    {
                        let key = highgui::wait_key(5)?;
                        if key > 0 && key != 255 {
                            timelapse.release()?;
                            break;
                        }
                    }
                }
                _ => {
                    error!("{}: Error reading frame from video stream", &self.name);
                    break;
                }
            }
        }
        Ok(())
    }
}

    // Write the frames in a separate thread
//    - doing this in the main thread causes stalls on the input stream
    fn async_writer(rx: Receiver<FrameSend>, video_filename: String, image_filename: String, fps: f64, fsize: Size, notify_end_person: bool, camera_name: String)
    {
        let rx = Arc::new(Mutex::new(rx));
        thread::spawn(move || {
            let rx = rx.lock().unwrap();
            let mut best_frame = Mat::default();
            let mut best_time = String::default();
            let mut have_best = false;

            let mut person_writer = create_video_writer(&video_filename, fps, fsize);
            loop
            {
                let r = rx.recv();
                if let Ok(r) = r {
                    match r {
                        FrameSend::Frame(f) => { person_writer.write(&f).unwrap(); }
                        FrameSend::Best(fm, timestamp) => {
                            best_frame = fm;
                            best_time = timestamp;
                            have_best = true;
                        }
                        FrameSend::End => { break; }
                    }
                }
            }

            person_writer.release().unwrap();

            // write the best frame
            let filename = format!("captures/people/photos/{}{}-best.jpg", camera_name, best_time);
            if have_best
            {
                let flags = Vector::new();
                imwrite(&filename, &best_frame, &flags);
            }

            info!("Person recording finished.");

            // Call the notifier
            if notify_end_person
            {
                let image = match have_best {
                    true => { filename }
                    false => { image_filename }
                };

                info!("Calling 'notify_end_person.sh {} {}'", &image, &video_filename);
                let r = Command::new("./notify_end_person.sh")
                    .arg(image).arg(&video_filename).spawn();
                if let Err(e) = r { error!("Error calling script: {}", e) }
            }
        });
    }


fn person_in_frame(int_mutex: &Arc<Mutex<Interpreter>>, frame320: &Mat, d: f32) -> Option<Rect>
{
    let mut interpreter = int_mutex.lock().unwrap();

    // Create input tensor
    let mut it = interpreter.inputs();
    let input_bytes = it[0].bytes_mut();

    // Copy pixel data swapping from opencv BGR format
    let mut o = 0;
    let src = frame320.data_bytes().unwrap();
    for _ in 1..RESOLUTION {
        for _ in 1..RESOLUTION {
            input_bytes[o + 0] = src[o + 2]; // R
            input_bytes[o + 1] = src[o + 1]; // G
            input_bytes[o + 2] = src[o + 0]; // B
            o = o + 3;
        }
    }

    // Raw copy also seems to work but is no faster (on a MacBook Pro).
    //input_bytes.copy_from_slice(frame320.data_bytes()?);

    let r = interpreter.invoke();
    match r {
        Err(_) => { error!("EdgeTPU invoke failed"); }
        _ => {}
    }

    let ot = interpreter.outputs();

    for i in 0..50
    {
        if ((ot[1].f32s()[i] + 1.0) as i32 == 1)  // Person class
            && (ot[2].f32s()[i] > THRESHOLD)
        {
            let x = (ot[0].f32s()[(i * 4) + 1] * (RESOLUTION as f32) * d) as i32;
            let y = (ot[0].f32s()[(i * 4) + 0] * (RESOLUTION as f32) * d) as i32;
            let w = (ot[0].f32s()[(i * 4) + 3] * (RESOLUTION as f32) * d) as i32;
            let h = (ot[0].f32s()[(i * 4) + 2] * (RESOLUTION as f32) * d) as i32;
            let r = Rect {
                x,
                y,
                width: w - x,
                height: h - y,
            };
            return Some(r);
        }
    }
    None
}


    fn create_video_writer(filename: &str, fps: f64, size: Size) -> VideoWriter
    {
        let fourcc = VideoWriter::fourcc('m' as i8, 'p' as i8, '4' as i8, 'v' as i8).expect("Invalid video fourcc");
        let writer = VideoWriter::new(&filename, fourcc, fps, size, true);
        match writer {
            Ok(writer) => {
                info!("Creating new video file: {}", filename);
                writer
            }
            Err(e) => {
                error!("Can't create video writer {}", filename);
                error!("Error: {}", e.message);
                panic!("{}", e);
            }
        }
    }


    fn inside_polygon(polygon: &Option<Vec<Point>>, point: &Point) -> bool
    {
        match polygon
        {
            Some(polygon) => {
                let mut inside = false;

                let mut j = polygon.last().unwrap();

                for i in polygon
                {
                    if (i.y < point.y) && (j.y >= point.y) || (j.y < point.y) && (i.y >= point.y)
                    {
                        if i.x + (point.y - i.y) / (j.y - i.y) * (j.x - i.x) < point.x
                        {
                            inside = !inside;
                        }
                    }
                    j = i;
                }

                inside
            },
            None => true
        }
    }

    fn draw_boundary(polygon: &Vec<Point>, frame: &mut Mat)
    {
        let mut last = None;
        let color = Scalar::from((128.0, 192.0, 192.0));

        for p in polygon
        {
            let p = opencv::core::Point::new(p.x, p.y);
            if let Some(l) = last
            {

                let r = line(frame, l, p.clone(), color, LINE_8, 0, 0);
                if let Err(_) = r { error!("Error drawing polygon line {:?} -> {:?}", l, p); }
            }
            last = Some(p.clone());
        }
    }

    fn timestamp_string() -> String
    {
        let local: DateTime<Local> = Local::now();
        local.format("%Y%m%d-%H%M%S").to_string()
    }