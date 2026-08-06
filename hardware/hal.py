"""Hardware abstraction layer: interfaces for every planned peripheral.

The device roadmap adds an ESP32-S3 co-processor, BNO085 IMU, BN-880 GPS,
MCP23017 I/O expander, MAX98357A audio amp + speaker, and status LEDs/buttons.
None of those are connected yet — but the *interfaces* exist now, so the day a
driver lands it plugs into `build_hardware()` and nothing above this layer
changes: the runtime talks to `HardwareSuite`, never to a chip.

Rules for implementations:
  * constructors must not raise on missing hardware — return unavailable state
    and report it through `health()` instead (a missing option is not an error);
  * everything here is optional: the vision product runs fully with all-null
    hardware (exactly today's Pi + camera build);
  * drivers live in hardware/drivers/<chip>.py when they arrive, and
    build_hardware() selects them by the `hardware:` config block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


# ------------------------------------------------------------------ interfaces
class IMU(ABC):
    """Orientation/motion sensing (BNO085). Future use: crash/harsh-event detection."""

    available = False

    @abstractmethod
    def read(self) -> Optional[dict]:
        """Latest sample: {"accel": (x,y,z), "gyro": (x,y,z), "quat": (w,x,y,z)} or None."""


class GPS(ABC):
    """Positioning (BN-880). Future use: trip logs, speed-aware alert policy."""

    available = False

    @abstractmethod
    def read(self) -> Optional[dict]:
        """Latest fix: {"lat": .., "lon": .., "speed_mps": .., "fix": bool} or None."""


class IOExpander(ABC):
    """Extra digital I/O (MCP23017) for buttons, LEDs, and relays beyond the Pi header."""

    available = False

    @abstractmethod
    def read_pin(self, pin: int) -> bool: ...

    @abstractmethod
    def write_pin(self, pin: int, value: bool) -> None: ...


class AudioOutput(ABC):
    """Speaker playback (MAX98357A I2S amp). Future use: voice alerts instead of tones."""

    available = False

    @abstractmethod
    def play(self, wav_path: str) -> None: ...


class LEDArray(ABC):
    """Status LEDs (direct GPIO or via the expander)."""

    available = False

    @abstractmethod
    def set(self, name: str, on: bool) -> None:
        """Set a named LED ("status", "alert", "provisioning") on or off."""


class ButtonArray(ABC):
    """Physical buttons; callers register callbacks by button name."""

    available = False

    @abstractmethod
    def on_press(self, name: str, callback: Callable[[], None]) -> None: ...


class ESP32Bridge(ABC):
    """Link to the ESP32-S3 co-processor (serial/USB). The ESP32 will own
    hard-real-time duties (IMU fusion, precise IO timing) and forward events."""

    available = False

    @abstractmethod
    def send(self, command: dict) -> None: ...

    @abstractmethod
    def poll(self) -> list[dict]:
        """Drain any pending messages from the co-processor."""


# ------------------------------------------------------------------ null implementations
class NullIMU(IMU):
    def read(self) -> Optional[dict]:
        return None


class NullGPS(GPS):
    def read(self) -> Optional[dict]:
        return None


class NullIOExpander(IOExpander):
    def read_pin(self, pin: int) -> bool:
        return False

    def write_pin(self, pin: int, value: bool) -> None:
        pass


class NullAudioOutput(AudioOutput):
    def play(self, wav_path: str) -> None:
        pass


class NullLEDArray(LEDArray):
    def set(self, name: str, on: bool) -> None:
        pass


class NullButtonArray(ButtonArray):
    def on_press(self, name: str, callback: Callable[[], None]) -> None:
        pass


class NullESP32Bridge(ESP32Bridge):
    def send(self, command: dict) -> None:
        pass

    def poll(self) -> list[dict]:
        return []


# ------------------------------------------------------------------ suite
@dataclass
class HardwareSuite:
    """Everything the runtime can touch, always fully populated (nulls when absent)."""

    imu: IMU = field(default_factory=NullIMU)
    gps: GPS = field(default_factory=NullGPS)
    io_expander: IOExpander = field(default_factory=NullIOExpander)
    audio: AudioOutput = field(default_factory=NullAudioOutput)
    leds: LEDArray = field(default_factory=NullLEDArray)
    buttons: ButtonArray = field(default_factory=NullButtonArray)
    esp32: ESP32Bridge = field(default_factory=NullESP32Bridge)

    def health_checks(self) -> dict[str, Callable[[], tuple[bool, str]]]:
        """Per-peripheral health checks for components configured as expected.

        Null (absent) peripherals report healthy-with-detail so the app's health
        card shows what's fitted without flagging missing options as failures.
        """
        def check_for(name: str, device) -> Callable[[], tuple[bool, str]]:
            return lambda: (True, "fitted") if device.available else (True, "not fitted")

        return {
            f"hw_{name}": check_for(name, device)
            for name, device in {
                "imu": self.imu, "gps": self.gps, "io_expander": self.io_expander,
                "audio": self.audio, "leds": self.leds, "buttons": self.buttons,
                "esp32": self.esp32,
            }.items()
        }


def build_hardware(hardware_cfg: dict) -> HardwareSuite:
    """Construct the suite from config.yaml's `hardware:` block.

    Today every entry resolves to the null implementation; when a driver lands,
    map its config key here (e.g. `imu: bno085` -> hardware.drivers.bno085.BNO085).
    Unknown/missing keys are never fatal.
    """
    return HardwareSuite()
