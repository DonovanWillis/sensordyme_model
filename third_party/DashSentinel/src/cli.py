"""all arguments for the command line interface (CLI) are defined here"""

import argparse

def parse_args():
    """parse command line arguments for the driver monitoring system"""

    parser = argparse.ArgumentParser(
        description="Adaptive driver monitor for edge devices"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=20)
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    parser.add_argument("--max-faces", type=int, default=3)
    parser.add_argument("--mirror", action="store_true")

    parser.add_argument(
        "--profile-path", type=str, default="./data/driver_profile.json"
    )
    parser.add_argument(
        "--log-path", type=str, default="./logs/dashsentinel_events.csv"
    )
    parser.add_argument("--log-csv", action="store_true")

    parser.add_argument("--calibration-seconds", type=int, default=25)
    parser.add_argument("--startup-baseline-seconds", type=int, default=12)
    parser.add_argument("--startup-baseline-min-frames", type=int, default=90)
    parser.add_argument("--rebuild-baseline-on-start", action="store_true")

    parser.add_argument("--ear-threshold", type=float, default=0.23)
    parser.add_argument("--yawn-mar-threshold", type=float, default=0.45)
    parser.add_argument("--yawn-frames-threshold", type=int, default=12)
    parser.add_argument("--warning-threshold", type=float, default=0.42)
    parser.add_argument("--drowsy-threshold", type=float, default=0.58)
    parser.add_argument("--status-hold-frames", type=int, default=10)
    parser.add_argument("--no-face-hold-frames", type=int, default=18)
    parser.add_argument("--low-quality-hold-frames", type=int, default=8)
    parser.add_argument("--attention-window", type=int, default=240)

    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--refine-landmarks", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")

    parser.add_argument("--min-frame-brightness", type=float, default=35.0)
    parser.add_argument("--max-frame-brightness", type=float, default=225.0)
    parser.add_argument("--min-frame-contrast", type=float, default=18.0)
    parser.add_argument("--min-frame-blur", type=float, default=45.0)

    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="optional local ONNX DMS classifier; no cloud calls are made",
    )
    parser.add_argument("--local-model-input-size", type=int, default=224)
    parser.add_argument("--min-model-confidence", type=float, default=0.55)
    parser.add_argument(
        "--require-local-model",
        action="store_true",
        help="fail fast if --local-model-path is missing or cannot be loaded",
    )

    parser.add_argument("--show-ui", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--window-name", type=str, default="DashSentinel")
    parser.add_argument("--save-profile-every-seconds", type=int, default=15)

    parser.add_argument(
        "--enable-esp-serial",
        action="store_true",
        help="send status and attentiveness to an esp8266/esp32 over usb serial",
    )
    parser.add_argument(
        "--esp-port",
        type=str,
        default="/dev/ttyUSB0",
        help="usb serial port for esp8266, e.g. /dev/ttyUSB0, /dev/ttyACM0, or COM3",
    )
    parser.add_argument(
        "--esp-baud", type=int, default=115200, help="serial baud rate for esp8266"
    )
    parser.add_argument(
        "--esp-send-interval",
        type=float,
        default=0.5,
        help="seconds between serial telemetry updates",
    )
    return parser.parse_args()
