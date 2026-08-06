"""
Module for handling serial telemetry communication to/from the esp8266
"""

import time
import serial

class SerialTelemetry:
    """handle serial communication with esp8266 for telemetry and button commands"""
    # default port can be changed with --serial-port,
    # baud rate can be changed with --serial-baud,
    # and send interval can be changed with --serial-interval
    def __init__(self, enabled=False, port="/dev/ttyUSB0", baud=115200, interval=0.5):
        self.enabled = enabled
        self.port = port
        self.baud = baud
        self.interval = interval
        self.last_send = 0.0
        self.serial = None

        if not self.enabled:
            return

        # attempt to open the serial port and connect to the esp8266
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2.0)
            print(f"[serial] connected to esp8266 on {self.port} at {self.baud} baud")
        except (serial.SerialException, OSError) as exc:
            self.serial = None
            print(f"[serial] could not open esp8266 serial port {self.port}: {exc}")

    def send(self, status, attentiveness, drowsy_score):
        """periodically send the stats to the esp"""
        if not self.enabled or self.serial is None:
            return

        now = time.time()
        if now - self.last_send < self.interval:
            return

        self.last_send = now

        try:
            # simple csv-style line to be parsed on esp8266
            msg = f"{status},{attentiveness:.1f},{drowsy_score:.3f}\n"
            self.serial.write(msg.encode("utf-8"))
            self.serial.flush()

        except serial.SerialException as exc:
            print(f"[serial] send failed: {exc}")
            try:
                self.serial.close()
            except serial.SerialException:
                pass
            self.serial = None

    def read_esp_command(self):
        """read any incoming commands sent from the esp8266 via serial"""
        if self.serial is None:
            return None

        if self.serial.in_waiting <= 0:
            return None

        line = self.serial.readline().decode("utf-8", errors="ignore").strip()

        # handle the button press commands sent from the esp8266
        if line == "BTN_RESET_BASELINE":
            return "RESET_BASELINE"

        if line == "BTN_RESET_STATS":
            return "RESET_STATS"

        return None

    def close(self):
        """gracefully close the serial connection"""
        if self.serial is not None:
            try:
                self.serial.close()
            except serial.SerialException:
                pass
            self.serial = None
