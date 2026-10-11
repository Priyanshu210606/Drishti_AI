# Phase 0 Report: UrbanEgo Dataset Inspection & Verification

## 1. Verified Dataset Inventory
- **Location:** `Data/` contains two recording sessions:
  - `Data/run7_20260714_171427/run7_20260714_171427/`
  - `Data/run8_20260714_174138/run8_20260714_174138/`
- **Origin:** Microsoft HoloLens 2 pedestrian recording dataset ("UrbanEgo", Aveiro, Portugal).

## 2. Sensor Stream Audit & Schemas
| Stream | Format | Rate / Frames | Key Fields | Verified Details |
|---|---|---|---|---|
| RGB Video | `rgb.mp4` | 20.0 fps | Video stream | 1280x720 H.264 video. Non-joinable MP4 frame index directly. |
| RGB Metadata | `rgb_frames.jsonl` | ~21.56 fps | `seq`, `ts_unix_ns`, `ts_mono_ns`, `pose`, `pv_focal_length`, `pv_principal_point`, `pv_resolution`, `frame` | Primary temporal frame anchor. Note: HoloLens `pose` entries are unpopulated (all 0.0). |
| Depth Images | `depth/*.png` | ~4.77 fps | PNG images | Resolution: 320x288, uint16 (`I;16`), unit: millimetres (mm). 0 = no return. |
| IR Images | `ab/*.png` | ~4.77 fps | PNG images | Resolution: 320x288, uint16 active brightness infrared. |
| Depth Metadata | `depth_frames.jsonl` | ~4.77 fps | `seq`, `ts_unix_ns`, `ts_mono_ns`, `pose`, `frame` | Joins to depth frames. `pose` is all 0.0. |
| Calibration | `calibration/` | Static | `calibration.json`, `ray_x.csv`, `ray_y.csv`, `ray_z.csv` | 288x320 float grids representing per-pixel unit rays. 3D coordinates: $X = \text{ray\_x} \times \frac{\text{depth}}{1000}$, $Y = \text{ray\_y} \times \frac{\text{depth}}{1000}$, $Z = \text{ray\_z} \times \frac{\text{depth}}{1000}$ (in metres). |
| IMU | `imu.jsonl` | ~14 Hz | `seq`, `ts_unix_ns`, `ts_mono_ns`, `yaw`, `pitch`, `roll` | Orientation Euler angles in degrees. Continuous head motion tracking. |
| Heading | `heading.jsonl` | ~14 Hz | `seq`, `ts_unix_ns`, `ts_mono_ns`, `heading` | Compass heading angle in degrees. Matches IMU yaw orientation. |
| GPS Phone | `gps_phone.jsonl` | ~1 Hz | `seq`, `ts_unix_ns`, `ts_mono_ns`, `latitude`, `longitude`, `altitude` | Geolocation fixes from mobile device. |
| GPS Receiver | `gps_receiver.jsonl` | ~1 Hz | `seq`, `ts_unix_ns`, `ts_mono_ns`, `latitude`, `longitude`, `altitude` | Geolocation fixes from external receiver. |
| YOLO Detections | `yolo/detections.jsonl` | Event-based | `t`, `ts_ns`, `frame`, `cls`, `name`, `id`, `conf`, `box` | Bounding boxes `[x1, y1, x2, y2]`, COCO classes (person, bicycle, vehicle, etc.), confidence score. |
| YOLO Summary | `yolo/frames.jsonl` | Frame-based | `frame`, `t`, `ts_ns`, `n`, `counts`, `per_class` | Aggregate obstacle counts per frame. |

## 3. Timestamp Synchronization
- Cross-stream synchronization is performed using `ts_unix_ns` (nanoseconds since epoch).
- YOLO detections align to RGB metadata via `ts_ns == ts_unix_ns` and matching 1-based `frame` indices.
- Depth and IMU streams align to the nearest temporal RGB frame without future leakage.
