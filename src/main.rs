mod config;
mod camera;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use docopt::Docopt;
use serde::Deserialize;

use opencv::{
    Result
};

use moonfire_tflite::*;
use crate::camera::{Camera, Point};
use crate::config::{CliConfig, Config, USAGE};

#[macro_use] extern crate log;


fn main() -> Result<()>
{
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

    let config: CliConfig = Docopt::new(USAGE)
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
    let mut polygon: Vec<Point> = Vec::default();
    if config.flag_polygon.is_some()
    {
        polygon = read_polygon_file(&config.flag_polygon.unwrap());
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

    // Wrap interpreter
    let interpreter = Arc::new(Mutex::new(interpreter));

    match config.flag_config
    {
        Some(f) => {
            let config = Config::load(&f).expect(&format!("Can't load config file {}", &f));

            info!("Config: {:?}", &config);

            let mut threads = vec![];

            for c in config.cameras
            {
                let interpreter = Arc::clone(&interpreter);
                threads.push(thread::spawn(move || {
                    loop {
                        if let Err(e) = c.run(Arc::clone(&interpreter), notify_start_person, notify_end_person, notify_timelapse_rollover)
                        {
                            error!("{}: {:?}", c.name, e);
                        }
                        info!("Camera \'{}\' disconnected, will reconnect in 10s...", &c.name);
                        sleep(Duration::from_secs(10));
                    }
                }));
            }

            threads.into_iter().for_each(|thread| {
                thread
                    .join()
                    .expect("The thread creating or execution failed !")
            });

        }
        None => {
            // Create Single Camera instance when no config file
            let mut camera = Camera::new(&config.arg_video_source);
            if config.flag_monitor { camera.monitor = true; }
            if config.flag_polygon.is_some()
            {
                camera.boundary = Some(read_polygon_file(&config.flag_polygon.unwrap()));
            }

            camera.run(interpreter, notify_start_person, notify_end_person, notify_timelapse_rollover)?;

        }
    }



    Ok(())
}


#[derive(Debug, Deserialize)]
struct CsvRecord {
    x: i32,
    y: i32,
}

fn read_polygon_file(filename: &str) -> Vec<Point>
{
    let mut polygon = Vec::new();
    let rdr = csv::Reader::from_path(Path::new(filename));
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