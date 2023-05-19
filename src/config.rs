use std::fs;
use std::path::Path;
use serde::Deserialize;
use crate::camera::Camera;

pub const USAGE: &'static str = "
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
  -c --config <config-file>         Use a config file (for multiple camera monitoring)
";


#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub cameras: Vec<Camera>
}

impl Config
{
    pub fn load(filename: &str) -> Result<Config, Box<dyn std::error::Error>>
    {
        let contents = fs::read_to_string(Path::new(filename))?;
        let config = serde_json::from_str(&contents)?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct CliConfig {
    pub arg_video_source: String,
    pub flag_monitor: bool,
    pub flag_timelapse: bool,
    pub flag_polygon: Option<String>,
    pub flag_config: Option<String>,
}

