mod config;

use std::time::SystemTime;
use std::path::Path;
use std::process::Command;
use chrono::{DateTime, Local, Timelike};
use docopt::Docopt;
use serde::Deserialize;

use opencv::{
    highgui,
    prelude::*,
    Result,
    videoio,
};

use moonfire_tflite::*;
use opencv::core::{Point, Rect, Scalar, Size, Vector};
use opencv::imgproc::{INTER_AREA, line, rectangle, resize};
use opencv::imgproc::LINE_8;
use opencv::imgcodecs::imwrite;
use opencv::videoio::{VideoCapture, VideoWriter};
use crate::config::{Config, USAGE};

#[macro_use] extern crate log;

const RESOLUTION: i32 = 320;  // input tensor resolution
const THRESHOLD: f32 = 0.6;

fn main() -> Result<()> {
    let window = "Security Camera";

    // Notification events
    let mut notify_start_person = false;
    let mut notify_end_person = false;
    let mut notify_timelapse_rollover = false;

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Info)
        .chain(std::io::stdout())
        //.chain(fern::log_file("camera.log").unwrap())
        .apply().expect("Can't initialise logging");

    let config: Config = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    // Check directories
    if !Path::new("captures/people/video").exists()
    {
        error!("'captures/people/video' directory does not exist at this location.");
        panic!("Unable to proceed");
    }
    if !Path::new("captures/people/photos").exists()
    {
        error!("'captures/people/photos' directory does not exist at this location.");
        panic!("Unable to proceed");
    }
    if config.flag_timelapse && !Path::new("captures/timelapse").exists()
    {
        error!("'captures/timelapse' directory does not exist at this location.");
        panic!("Unable to proceed");
    }

    // Check notify scripts
    if Path::new("notify_start_person.sh").exists()
    {
        info!("'notify_start_person.sh <first-image-file>' will be called.");
        notify_start_person = true;
    }
    if Path::new("notify_end_person.sh").exists()
    {
        info!("'notify_end_person.sh <best-image-file> <video-file>' will be called.");
        notify_end_person = true;
    }
    if config.flag_timelapse && Path::new("notify_timelapse_rollover.sh").exists()
    {
        info!("'notify_timelapse_rollover.sh <video-file>' will be called.");
        notify_timelapse_rollover = true;
    }

    // Boundary polygon
    let mut use_polygon = false;
    let mut polygon: Vec<Point> = Vec::default();
    if config.flag_polygon.is_some()
    {
        polygon = read_polygon_file(&config.flag_polygon.unwrap());
        use_polygon = true;
    }


    // Moonfire-tflite
    static EDGETPU_MODEL: &'static [u8] = include_bytes!("../ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite");
    let m = Model::from_static(EDGETPU_MODEL).unwrap();
    let mut builder = Interpreter::builder();

    // Configure EdgeTPU device
    let devices = edgetpu::Devices::list();
    if devices.is_empty() {
        error!("Can't find EdgeTPU device.");
        panic!("need an edge tpu installed to run edge tpu tests");
    } else {
        for d in &devices
        {
            info!("Using EdgeTPU device: {:?}", d);
            let delegate = d.create_delegate().unwrap();
            builder.add_owned_delegate(delegate);
        }
    }

    let mut interpreter = builder.build(&m).unwrap();
    info!(
        "Successfully create tflite interpreter with {} inputs, {} outputs",
        interpreter.inputs().len(),
        interpreter.outputs().len()
    );

    if config.flag_monitor
    {
        highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
        info!("Opened monitor window");
    }


    let mut cam: VideoCapture;

    #[cfg(ocvrs_opencv_branch_32)]
    if config.arg_video_source.len() == 0
    {
        cam = videoio::VideoCapture::new_default(0)?;
        info!("Opening default video stream.");
    }
    else {
        cam = videoio::VideoCapture::open(&config.arg_video_source);
        info!("Opening video stream at {}", config.arg_video_source);
    }

    #[cfg(not(ocvrs_opencv_branch_32))]
    if config.arg_video_source.len() == 0
    {
        cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
        info!("Opening default video stream.");
    }
    else {
        cam = videoio::VideoCapture::from_file(&config.arg_video_source, 0)?; // stream
        info!("Opening video stream at {}", config.arg_video_source);
    }

    let opened = videoio::VideoCapture::is_opened(&cam)?;

    if !opened {
        error!("Can't open video stream !");
        panic!("Unable to proceed");
    }

    // Initialisation
    let mut frame = Mat::default();
    let mut frame320 = Mat::default();
    let size320= Size::new(RESOLUTION, RESOLUTION);
    let mut fx: i32 = 0;
    let mut fy: i32 = 0;
    let mut fw = 0;
    let mut d = 0.0;
    let mut fsize = Size::new(0,0);

    // Get video stream size - input tensor is 320 x 320 RGB so we'll take the central square from the video stream
    cam.read(&mut frame)?;
    if frame.size()?.width > 0 {
        let w = frame.size()?.width;
        let h = frame.size()?.height;
        d = (w as f32) / (RESOLUTION as f32);
        let dy = (h as f32) / (RESOLUTION as f32);
        if dy < d { d = dy; }
        fx = (w - (((RESOLUTION as f32) * d) as i32)) / 2;
        fy = (h - (((RESOLUTION as f32) * d) as i32)) / 2;
        fw = fx + (((RESOLUTION as f32) * d) as i32);
        fsize = frame.size()?.clone();
    }

    let mut tick = SystemTime::now();
    let mut frames = 0;
    let mut fps = 0.0;

    // Timelapse recording
    let mut timelapse_filename= "".to_string();
    let mut timelapse: VideoWriter = VideoWriter::default()?;
    let mut skip_timlapse = false;
    if config.flag_timelapse
    {
        info!("Timelapse recording is enabled.");
        timelapse_filename = format!("captures/timelapse/{}.mp4", timestamp_string());
        timelapse = create_video_writer(&timelapse_filename, 1.5, fsize);
    }
    else { info!("Timelapse recording is disabled."); }

    // Person recording
    let mut buffer_size = 150;
    let mut buffer_pnt = 0;
    let mut buffer: Vec<Mat> = Vec::with_capacity(buffer_size);
    let mut person_recording = false;
    let mut person_best_frame = Mat::default();
    let mut person_best_size = 0;
    let mut person_best_time = "".to_string();
    let mut person_last_seen = SystemTime::now();
    let mut person_video_file = "".to_string();
    let mut person_writer: VideoWriter = VideoWriter::default()?;

    // Main activity loop
    loop {
        let rs = cam.read(&mut frame);
        match rs {
            Ok(true) => {
                if frame.size()?.width > 0 {
                    let mut frame320rc = frame.col_bounds(fx, fw);
                    match frame320rc {
                        Ok(mut frame320rc) => {
                            resize(&frame320rc, &mut frame320, size320, 0.0, 0.0, INTER_AREA);

                            // Create input tensor
                            let mut it = interpreter.inputs();
                            let mut input_bytes = it[0].bytes_mut();

                            // Copy pixel data swapping from opencv BGR format
                            let mut o = 0;
                            let src = frame320.data_bytes()?;
                            for y in 1..RESOLUTION {
                                for x in 1..RESOLUTION {
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
                                Err(e) => { println!("Invoke failed"); }
                                _ => {}
                            }

                            let ot = interpreter.outputs();

                            for i in 0..50
                            {
                                if (ot[2].f32s()[i] > THRESHOLD)
                                {
                                    let x = (ot[0].f32s()[(i*4) + 1] * (RESOLUTION as f32) * d) as i32;
                                    let y = (ot[0].f32s()[(i*4) + 0] * (RESOLUTION as f32)* d) as i32;
                                    let w = (ot[0].f32s()[(i*4) + 3] * (RESOLUTION as f32)* d) as i32;
                                    let h = (ot[0].f32s()[(i*4) + 2] * (RESOLUTION as f32)* d) as i32;
                                    let r = Rect {
                                        x,
                                        y,
                                        width: w - x,
                                        height: h - y
                                    };

                                    let object = (ot[1].f32s()[i] + 1.0) as i32;
                                    if object == 1 // Person
                                    {
                                        let outside_color = Scalar::from((64.0, 64.0, 240.0));
                                        let inside_color = Scalar::from((64.0, 240.0, 64.0));

                                        let centre = Point::new(r.x + r.width / 2, r.y + r.height / 2);

                                        if !use_polygon || inside_polygon(&polygon, &centre)
                                        {
                                            if config.flag_monitor
                                            {
                                                rectangle(&mut frame320rc, r, inside_color, 2, LINE_8, 0);
                                            }

                                            person_last_seen = SystemTime::now();
                                            let area = r.height * r.width;
                                            if area > person_best_size
                                            {
                                                person_best_size = area;
                                                person_best_frame = frame.clone();
                                                person_best_time = timestamp_string();
                                            }

                                            if (!person_recording)
                                            {
                                                // Start recording
                                                info!("Person detected - recording started");
                                                person_recording = true;
                                                person_video_file = format!("captures/people/video/{}.mp4", timestamp_string());
                                                person_writer = create_video_writer(&person_video_file, fps, fsize);

                                                // Write the buffer frames
                                                for i in buffer_pnt..buffer_size
                                                {
                                                    let f = buffer.get(i);
                                                    if let Some(frame) = f { person_writer.write(frame)?; }
                                                }
                                                if buffer_pnt > 0 {
                                                    for i in 0..(buffer_pnt-1)
                                                    {
                                                        let f = buffer.get(i);
                                                        if let Some(frame) = f { person_writer.write(frame)?; }
                                                    }
                                                }
                                                buffer = Vec::with_capacity(buffer_size);
                                                buffer_pnt = 0;

                                                // Write first photo and call notifier
                                                let flags = Vector::new();
                                                let filename = format!("captures/people/photos/{}-first.jpg", timestamp_string());
                                                imwrite(&filename, &frame, &flags);
                                                if notify_start_person
                                                {
                                                    info!("Calling 'notify_start_person.sh {}'", &filename);
                                                    let r = Command::new("./notify_start_person.sh")
                                                        .arg(filename).spawn();
                                                    if let Err(e) = r { error!("Error calling script: {}", e) }
                                                }
                                            }
                                        }
                                        else {
                                            if config.flag_monitor
                                            {
                                                rectangle(&mut frame320rc, r, outside_color, 2, LINE_8, 0);
                                            }
                                        }
                                    }
                                }
                            }

                            // Person recording
                            if person_recording
                            {
                                person_writer.write(&frame)?;
                                let elapsed = SystemTime::now().duration_since(person_last_seen).unwrap().as_millis();
                                if elapsed > 30000   // 30 seconds since last activity
                                {
                                    person_writer.release()?;

                                    let flags = Vector::new();
                                    let best_filename = format!("captures/people/photos/{}-best.jpg", person_best_time);
                                    imwrite(&best_filename, &person_best_frame, &flags);
                                    person_best_size = 0;

                                    person_recording = false;
                                    info!("Person recording finished.");

                                    // Call the notifier
                                    if notify_end_person
                                    {
                                        info!("Calling 'notify_end_person.sh {} {}'", &best_filename, &person_video_file);
                                        let r = Command::new("./notify_end_person.sh")
                                            .arg(best_filename).arg(&person_video_file).spawn();
                                        if let Err(e) = r { error!("Error calling script: {}", e) }
                                    }
                                }
                            }
                            else
                            {
                                // Stash the frame in the buffer
                                if buffer.len() <= buffer_pnt { buffer.push(frame.clone()); }
                                else { buffer[buffer_pnt] = frame.clone(); }
                                buffer_pnt = buffer_pnt + 1;
                                if buffer_pnt == buffer_size { buffer_pnt = 0 };
                            }

                            // Time related
                            frames += 1;

                            let elapsed = SystemTime::now().duration_since(tick).unwrap().as_millis();
                            if elapsed > 1000
                            {
                                // Update fps
                                fps = (frames as f64) / ((elapsed as f64) / 1000.0);
                                frames = 0;
                                tick = SystemTime::now();

                                if config.flag_timelapse
                                {
                                    // Write timelapse frame
                                    timelapse.write(&frame);

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

                                        timelapse_filename = format!("captures/timelapse/{}.mp4", timestamp_string());
                                        timelapse = create_video_writer(&timelapse_filename,1.5, fsize);

                                    } else { skip_timlapse = false; }
                                }
                            }

                            if config.flag_monitor
                            {
                                if use_polygon { draw_boundary(&polygon, &mut frame320rc); }
                                highgui::imshow(window, &mut frame)?;
                            }
                        }
                        Err(e) => { error!("Error extracting columns from frame: {}", e); }
                    }
                }
                if config.flag_monitor
                {
                    let key = highgui::wait_key(5)?;
                    if key > 0 && key != 255 {
                        timelapse.release()?;
                        break;
                    }
                }
            }
            _ => { error!("Error reading frame from video stream"); }
        }
    }
    Ok(())
}

fn timestamp_string() -> String
{
    let local: DateTime<Local> = Local::now();
    local.format("%Y%m%d-%H%M%S").to_string()
}

fn create_video_writer(filename: &str, fps: f64, size: Size) -> VideoWriter
{
    let fourcc = VideoWriter::fourcc('m' as u8,'p' as u8,'4' as u8,'v' as u8).expect("Invalid video fourcc");
    let mut writer = VideoWriter::new(&filename, fourcc, fps, size, true);
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

fn draw_boundary(polygon: &Vec<Point>, frame: &mut Mat)
{
   
    let mut last = None;
    let color = Scalar::from((128.0, 192.0, 192.0));

    for p in polygon
    {
        if let Some(l) = last
        {
            line(frame, l, *p, color, LINE_8, 0, 0);
        }
        last = Some(p.clone());
    }
}

pub fn inside_polygon(polygon: &Vec<Point>, point: &Point) -> bool
{
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
}

#[derive(Debug, Deserialize)]
struct CsvRecord {
    x: i32,
    y: i32,
}

fn read_polygon_file(filename: &str) -> Vec<Point>
{
    let mut polygon = Vec::new();
    let mut rdr = csv::Reader::from_path(Path::new(filename));
    match rdr {
        Ok(mut rdr) => {
            for result in rdr.deserialize() {
                match result {
                    Ok(result) => {
                        let record: CsvRecord = result;
                        polygon.push(Point::new(record.x, record.y));
                    }
                    Err(e) => {
                        error!("Error in polygon file {}: {}", filename, e);
                        panic!("Unable to proceed");
                    }
                }
            }
        }
        Err(e) => {
            error!("Can't read polygon file {}: {}", filename, e);
            panic!("Unable to proceed");
        }
    }
    info!("Read polygon file {} containing {:} points.", filename, polygon.len());
    polygon
}