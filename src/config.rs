use serde::Deserialize;

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
";


#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub arg_video_source: String,
    pub flag_monitor: bool,
    pub flag_timelapse: bool,
    pub flag_polygon: Option<String>,
}

pub fn default_config() -> Config
{
    Config {
        arg_video_source: "".to_string(),
        flag_monitor: false,
        flag_timelapse: false,
        flag_polygon: None,
    }
}