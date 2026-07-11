#!/usr/bin/env python3
"""
desk_pet.py

An interactive command-line "Desk Pet" application for LEGO WeDo 2.0.
Integrates with the global 'robot_midi' library to play songs through the Smarthub speaker
and animate the pet's motor/LED in sync with the music.

Features:
- Pure CLI menu and live-monitoring terminal dashboard using 'rich'.
- Autonomous state machine (sleeping, awake, happy, angry, dizzy, eating, singing).
- Non-blocking WeDo 2.0 sensor polling with notification-based caching.
- Integrated MIDI music player that searches BitMIDI and plays.
- Mock/Simulation fallback mode when hardware is not present.
"""

import sys
import os
import time
import json
import random
import threading
import argparse
import asyncio
import re
import select

# Add Documents and package directory to sys.path to ensure we can import robot_midi
sys.path.append("/home/zaiah/Documents")
sys.path.append("/home/zaiah/.local/lib/python3.12/site-packages")

try:
    import robot_midi
except ImportError:
    robot_midi = None

# Import rich library components
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import ProgressBar
    from rich.live import Live
    from rich.table import Table
    from rich.align import Align
except ImportError:
    print("Error: The 'rich' library is required to run this CLI dashboard.")
    print("Please install it: pip install rich")
    sys.exit(1)

# Import Bleak for WeDo 2.0 BLE connection
try:
    from bleak import BleakClient, BleakScanner
    ble_available = True
except ImportError:
    ble_available = False

# WeDo 2.0 UUIDs (standard LEGO BLE protocol)
WEDO_SERVICE_UUID    = "23d1bcea-5f78-2315-deef-1212000e4f00"
WEDO_OUTPUT_UUID     = "00001565-1212-efde-1523-785feabcd123"
WEDO_INPUT_UUID      = "00001563-1212-efde-1523-785feabcd123"
WEDO_SENSOR_UUID     = "00001560-1212-efde-1523-785feabcd123"
WEDO_NAME_UUID       = "00001524-1212-efde-1523-785feabcd123"
WEDO_BUTTON_UUID     = "00001526-1212-efde-1523-785feabcd123"
WEDO_BATTERY_UUID    = "00002a19-0000-1000-8000-00805f9b34fb"
WEDO_PORT_UUID       = "00001527-1212-efde-1523-785feabcd123"
WEDO_TURNOFF_UUID    = "0000152b-1212-efde-1523-785feabcd123"
WEDO_DISCONNECT_UUID = "0000152e-1212-efde-1523-785feabcd123"

# Sensor lookup dicts
DISTANCE_LOOKUP = {
    "2041": 10, "1041": 9, "0041": 8, "e040": 7, "c040": 6,
    "a040": 5, "8040": 4, "4040": 3, "0040": 2, "803f": 1,
    "0000": 0, "": 0
}

TILT_LOOKUP = {
    "0000": "Neutral", "4040": "Backward", "a040": "Right",
    "e040": "Left", "1041": "Forward", "2041": "Unknown", "": "Unknown"
}

WEDO_SENSORS = {
    b'\x01\x01#\x00\x00\x00\x10\x00\x00\x00\x10': "Distance Sensor",
    b'\x01\x00#\x00\x00\x00\x10\x00\x00\x00\x10': "Distance Sensor",
    b'\x01\x01"\x00\x00\x00\x10\x00\x00\x00\x10': "Tilt Sensor",
    b'\x01\x00"\x00\x00\x00\x10\x00\x00\x00\x10': "Tilt Sensor",
    b'\x01\x00\x01\x01\x00\x00\x00\x01\x00\x00\x00': "Motor",
    b'\x01\x01\x01\x01\x00\x00\x00\x01\x00\x00\x00': "Motor",
    b'\x00': "_"
}

CACHE_FILE = os.path.expanduser("~/.wedo_mac_cache.json")

# Console instance
console = Console()

# -----------------------------------------------------------------
# Bluetooth Connection Caching Utility
# -----------------------------------------------------------------
def get_cached_address(hub_name):
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f).get(hub_name)
    except Exception:
        pass
    return None

def set_cached_address(hub_name, address):
    try:
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        cache[hub_name] = address
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

# -----------------------------------------------------------------
# Non-blocking Keyboard Input for Linux Terminals
# -----------------------------------------------------------------
def is_key_pressed():
    r, _, _ = select.select([sys.stdin], [], [], 0.05)
    return len(r) > 0

def get_key():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# -----------------------------------------------------------------
# Async Helper Runner for Bleak
# -----------------------------------------------------------------
class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

runner = AsyncRunner()

# -----------------------------------------------------------------
# Smart Connection API: WeDo2Hub Implementation
# -----------------------------------------------------------------
class RealWeDo2Hub:
    def __init__(self, name: str, timeout: float = 6.0):
        self.hub_name = name
        self.client = None
        self.connected_state = False
        self.button_state = 0
        self.battery_level = 0
        self.port_data = ["_", "_"]
        
        # Sensor readings in memory to avoid BLE block during playback
        self.sensor_cache = {
            1: {"distance": 0, "tilt": "Neutral", "times": 0},
            2: {"distance": 0, "tilt": "Neutral", "times": 0}
        }

        # Try connecting with cached address
        address = get_cached_address(name)
        if address:
            try:
                self.client = BleakClient(address, disconnected_callback=self._on_disconnected)
                runner.run(asyncio.wait_for(self.client.connect(), timeout=3.0))
                self.connected_state = True
            except Exception:
                self.client = None

        if not self.client:
            # Scanning fallback
            device = runner.run(BleakScanner.find_device_by_name(name, timeout))
            if device is None:
                raise TimeoutError(f"LEGO Smarthub '{name}' not found.")
            self.client = BleakClient(device.address, disconnected_callback=self._on_disconnected)
            runner.run(self.client.connect())
            self.connected_state = True
            set_cached_address(name, device.address)

        time.sleep(0.5)
        self._register_notifications()
        time.sleep(0.3)
        self.get_battery_level()

    def _register_notifications(self):
        runner.run(self.client.start_notify(WEDO_BUTTON_UUID, self._notification_handler_button))
        runner.run(self.client.start_notify(WEDO_PORT_UUID, self._notification_handler_port))
        runner.run(self.client.start_notify(WEDO_SENSOR_UUID, self._notification_handler_sensor))

    def _on_disconnected(self, client):
        self.connected_state = False

    def _notification_handler_button(self, sender, data):
        if len(data) > 0:
            self.button_state = data[0]

    def _notification_handler_port(self, sender, data):
        try:
            port_idx = {b'\x01': 0, b'\x02': 1}[bytes(data[:1])]
            device_name = WEDO_SENSORS[bytes(data[1:])]
            self.port_data[port_idx] = device_name
            
            port = port_idx + 1
            if device_name == "Distance Sensor":
                self.write_raw_input_async(b'\x01\x02' + bytes([port]) + b'\x23\x00\x01\x00\x00\x00\x02\x01')
            elif device_name == "Tilt Sensor":
                self.write_raw_input_async(b'\x01\x02' + bytes([port]) + b'\x22\x01\x01\x00\x00\x00\x02\x01')
        except Exception:
            pass

    def _notification_handler_sensor(self, sender, data):
        try:
            if len(data) >= 2:
                port = data[1]
                val_hex = data[4:6].hex() if len(data) >= 6 else ""
                
                if port in self.sensor_cache:
                    self.sensor_cache[port]["times"] = data[0] if len(data) > 0 else 0
                    device = self.port_data[port - 1] if port <= len(self.port_data) else None
                    if device == "Distance Sensor":
                        self.sensor_cache[port]["distance"] = DISTANCE_LOOKUP.get(val_hex, 0)
                    elif device == "Tilt Sensor":
                        self.sensor_cache[port]["tilt"] = TILT_LOOKUP.get(val_hex, "Unknown")
        except Exception:
            pass

    def check_connected(self, device):
        if len(self.port_data) > 0 and self.port_data[0] == device:
            return 1
        if len(self.port_data) > 1 and self.port_data[1] == device:
            return 2
        return None

    def disconnect(self):
        self.connected_state = False
        if self.client:
            try:
                runner.run(asyncio.wait_for(self.client.write_gatt_char(WEDO_DISCONNECT_UUID, b'\x01'), timeout=1.0))
            except Exception:
                pass
            try:
                runner.run(self.client.disconnect())
            except Exception:
                pass

    def shut_off(self):
        self.connected_state = False
        if self.client:
            try:
                runner.run(asyncio.wait_for(self.client.write_gatt_char(WEDO_TURNOFF_UUID, b'\x01'), timeout=1.0))
            except Exception:
                pass

    def write_raw_output(self, command: bytes):
        if not self.connected_state or not self.client:
            return
        try:
            runner.run(asyncio.wait_for(self.client.write_gatt_char(WEDO_OUTPUT_UUID, command), timeout=1.0))
        except Exception:
            pass

    def write_raw_input_async(self, command: bytes):
        if not self.connected_state or not self.client:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.client.write_gatt_char(WEDO_INPUT_UUID, command), runner.loop)
        except Exception:
            pass

    def get_battery_level(self):
        if not self.connected_state or not self.client:
            return self.battery_level
        try:
            val = runner.run(asyncio.wait_for(self.client.read_gatt_char(WEDO_BATTERY_UUID), timeout=1.5))
            self.battery_level = int.from_bytes(val, 'little')
            return self.battery_level
        except Exception:
            return self.battery_level

    # Actuator wrapper methods
    def set_led(self, color_name_or_int):
        color_map = {
            "off": 0, "pink": 1, "purple": 2, "blue": 3, "sky blue": 4, 
            "teal": 5, "green": 6, "yellow": 7, "orange": 8, "red": 9, "white": 10
        }
        val = color_map.get(str(color_name_or_int).lower(), 0) if isinstance(color_name_or_int, str) else color_name_or_int
        self.write_raw_input_async(b'\x01\x02\x06\x17\x00\x01\x00\x00\x00\x02\x01')
        self.write_raw_output(bytes([0x06, 0x04, 0x01, val]))

    def set_led_rgb(self, r, g, b):
        self.write_raw_input_async(b'\x01\x02\x06\x17\x01\x01\x00\x00\x00\x02\x01')
        self.write_raw_output(bytes([0x06, 0x04, 0x03, r, g, b]))

    def set_motor(self, speed: int):
        port = self.check_connected("Motor")
        if not port:
            return
        speed = max(-100, min(100, speed)) & 0xFF
        self.write_raw_output(bytes([port, 0x01, 0x01, speed]))

    def stop_motor(self):
        port = self.check_connected("Motor")
        if not port:
            return
        self.write_raw_output(bytes([port, 0x01, 0x01, 0]))

    def beep(self, freq: int, duration_ms: int):
        freq_b = freq.to_bytes(2, 'little')
        dur_b  = int(duration_ms).to_bytes(2, 'little')
        self.write_raw_output(b'\x05\x02\x04' + freq_b + dur_b)

# -----------------------------------------------------------------
# Mock WeDo 2.0 Hub for Offline/Simulation Testing
# -----------------------------------------------------------------
class MockWeDo2Hub:
    def __init__(self, name: str):
        self.hub_name = name + " (MOCK)"
        self.connected_state = True
        self.button_state = 0
        self.battery_level = 94
        self.port_data = ["Distance Sensor", "Motor"]
        
        self.sensor_cache = {
            1: {"distance": 10, "tilt": "Neutral", "times": 0},
            2: {"distance": 0, "tilt": "Neutral", "times": 0}
        }
        self.current_led = "blue"
        self.motor_speed = 0

        # Run background mock drift thread
        self._drift_thread = threading.Thread(target=self._simulate_environment, daemon=True)
        self._drift_thread.start()

    def _simulate_environment(self):
        # Simulate slight shifts in distance & occasional tilt changes to show activity
        while self.connected_state:
            time.sleep(random.uniform(2.0, 5.0))
            if random.random() < 0.2:
                # Random obstacle approach
                for d in range(10, random.randint(1, 8), -1):
                    self.sensor_cache[1]["distance"] = d
                    time.sleep(0.2)
                time.sleep(random.uniform(1.0, 3.0))
                for d in range(self.sensor_cache[1]["distance"], 11):
                    self.sensor_cache[1]["distance"] = d
                    time.sleep(0.2)
            if random.random() < 0.15:
                # Random tilt
                tilts = ["Neutral", "Neutral", "Left", "Right", "Forward", "Backward"]
                self.sensor_cache[1]["tilt"] = random.choice(tilts)
                time.sleep(random.uniform(1.5, 4.0))
                self.sensor_cache[1]["tilt"] = "Neutral"

    def check_connected(self, device):
        if len(self.port_data) > 0 and self.port_data[0] == device:
            return 1
        if len(self.port_data) > 1 and self.port_data[1] == device:
            return 2
        return None

    def disconnect(self):
        self.connected_state = False

    def shut_off(self):
        self.connected_state = False

    def get_battery_level(self):
        # Drain battery slowly
        if random.random() < 0.02:
            self.battery_level = max(0, self.battery_level - 1)
        return self.battery_level

    def set_led(self, color_name_or_int):
        self.current_led = color_name_or_int

    def set_led_rgb(self, r, g, b):
        self.current_led = f"RGB({r},{g},{b})"

    def set_motor(self, speed: int):
        self.motor_speed = speed

    def stop_motor(self):
        self.motor_speed = 0

    def beep(self, freq: int, duration_ms: int):
        pass # mock beep is silent (or could use system bell)

# -----------------------------------------------------------------
# WeDo Music Player Adapter for robot_midi
# -----------------------------------------------------------------
class WeDoRobotAdapter:
    def __init__(self, hub, animate_cb=None):
        self.hub = hub
        self.animate_cb = animate_cb
        self.note_count = 0

    def beep(self, freq_hz: int, duration_ms: int):
        try:
            self.hub.beep(freq_hz, duration_ms)
            if self.animate_cb:
                self.note_count += 1
                self.animate_cb(freq_hz, duration_ms, self.note_count)
        except Exception:
            pass
        # Sleep for sound duration outside any lock to ensure smooth playback
        time.sleep(duration_ms / 1000.0)

# -----------------------------------------------------------------
# Desk Pet State Machine & Brain
# -----------------------------------------------------------------
class DeskPet:
    def __init__(self, hub, state_dict):
        self.hub = hub
        self.mood = "awake" # awake, sleeping, happy, angry, dizzy, eating, singing
        self.frame = 0
        
        # Vital stats (0 - 100)
        self.energy = state_dict.get("energy", 80)
        self.happiness = state_dict.get("happiness", 70)
        self.hunger = state_dict.get("hunger", 30) # lower is better (0 = satisfied, 100 = starving)
        
        # Profile settings
        self.profile = state_dict.get("profile", "Puppy")
        self.pet_name = state_dict.get("pet_name", "Kepler")
        
        # Leveling & XP
        self.level = state_dict.get("level", 1)
        self.xp = state_dict.get("xp", 0)
        
        self.logs = []
        self.log_lock = threading.Lock()
        self.is_running = True
        self.music_playing = False
        
        # Tail wagging properties
        self.tail_active = False
        
        # Ollama Autopilot Mode
        self.ai_autopilot = False
        self.last_ai_trigger_time = 0.0

        # Screaming feeding challenge
        self.screaming_for_food = False
        self.screaming_cycle_start = 0.0

        # Triple press history
        self.button_releases = []

        # Trainer Health & sleep covers
        self.trainer_hp = state_dict.get("trainer_hp", 100)
        self.sleep_cover_start = 0.0

        self.add_log(f"Pet {self.pet_name} initialized. Welcome!")

        # Start autonomous behavior thread
        self.brain_thread = threading.Thread(target=self._brain_loop, daemon=True)
        self.brain_thread.start()


    def add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        with self.log_lock:
            self.logs.append(f"[{timestamp}] {msg}")
            if len(self.logs) > 30:
                self.logs.pop(0)

    # -------------------------------------------------------------
    # State Persistence (Save/Load)
    # -------------------------------------------------------------
    def get_state_file_path(self):
        return os.path.expanduser(f"~/.wedo_pets/{self.pet_name}.json")

    def save_state(self):
        try:
            os.makedirs(os.path.expanduser("~/.wedo_pets"), exist_ok=True)
            state = {
                "pet_name": self.pet_name,
                "profile": self.profile,
                "level": self.level,
                "xp": self.xp,
                "energy": self.energy,
                "happiness": self.happiness,
                "hunger": self.hunger,
                "trainer_hp": self.trainer_hp
            }
            with open(self.get_state_file_path(), "w") as f:
                json.dump(state, f)
        except Exception as e:
            self.add_log(f"Error saving state: {e}")

    def load_state(self):
        path = self.get_state_file_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                    self.pet_name = state.get("pet_name", self.pet_name)
                    self.profile = state.get("profile", self.profile)
                    self.level = state.get("level", self.level)
                    self.xp = state.get("xp", self.xp)
                    self.energy = state.get("energy", self.energy)
                    self.happiness = state.get("happiness", self.happiness)
                    self.hunger = state.get("hunger", self.hunger)
                    self.trainer_hp = state.get("trainer_hp", self.trainer_hp)
            except Exception as e:
                self.add_log(f"Error loading state: {e}")


    # -------------------------------------------------------------
    # Leveling & XP System
    # -------------------------------------------------------------
    def get_level_title(self):
        titles = {
            1: "Newborn 🥚",
            2: "Toddler 🐾",
            3: "Explorer 🧭",
            4: "Companion 🤝",
            5: "Super Pet ⚡",
            6: "Legend 🌟"
        }
        return titles.get(self.level, "Ultimate Cosmic Companion 🌌")

    def get_xp_needed(self):
        return int((self.level ** 1.5) * 250)

    def gain_xp(self, amount):
        self.xp += amount
        if self.xp < 0:
            self.xp = 0
        xp_needed = self.get_xp_needed()
        if self.xp >= xp_needed:
            self.level_up()
        else:
            self.save_state()

    def level_up(self):
        xp_needed = self.get_xp_needed()
        self.xp = max(0, self.xp - xp_needed)
        self.level += 1
        self.add_log(f"★ LEVEL UP! ★ {self.pet_name} reached Level {self.level} ({self.get_level_title()})!")

        
        # Ascending chime beeps
        def play_level_chimes():
            chimes = [440, 554, 659, 880]
            for tone in chimes:
                try:
                    self.hub.beep(tone, 150)
                except Exception:
                    pass
                time.sleep(0.08)
        threading.Thread(target=play_level_chimes, daemon=True).start()
        self.save_state()

    # -------------------------------------------------------------
    # Ollama AI Autopilot & Chat Integration
    # -------------------------------------------------------------
    def get_soul_file_path(self):
        return os.path.expanduser(f"~/.wedo_pets/{self.pet_name}_soul.txt")


    def ensure_soul_file(self):
        path = self.get_soul_file_path()
        if not os.path.exists(path):
            try:
                default_soul = (
                    f"You are the personality and soul of {self.pet_name}, a loyal, playful, and friendly robot companion. "
                    "You love eating cookies, playing synthesized MIDI songs, and wagging your motor-controlled physical tail. "
                    "You react to distance and tilt changes. You learn and adapt to how the user treats you."
                )
                with open(path, "w") as f:
                    f.write(default_soul)
            except Exception as e:
                self.add_log(f"Error creating default soul file: {e}")

    def query_ollama(self, prompt):
        import urllib.request
        
        self.ensure_soul_file()
        soul_content = "You are a friendly robotic companion."
        try:
            with open(self.get_soul_file_path(), "r") as f:
                soul_content = f.read().strip()
        except Exception as e:
            self.add_log(f"Error reading soul file: {e}")

        url = "http://localhost:11434/api/generate"
        system_rules = (
            "You are the brain of an interactive desk pet robot built with LEGO WeDo 2.0.\n"
            f"Your name is {self.pet_name} and you are a {self.profile}.\n"
            f"Personality/Soul Context:\n{soul_content}\n\n"
            "You respond to user chat messages, sensor events, dreaming logs, or idle thoughts. "
            "You must respond ONLY with a valid JSON object matching exactly this schema:\n"
            "{\n"
            "  \"thought\": \"Brief internal thought (max 10 words)\",\n"
            "  \"speech\": \"Cute sound words or conversational response (max 40 words). Set to null if you decide to wait/do nothing.\",\n"
            "  \"emotion\": \"awake|sleeping|happy|angry|dizzy|eating|singing\",\n"
            "  \"color\": \"red|green|blue|yellow|purple|cyan|white|off\",\n"
            "  \"sound\": [[frequency_hz, duration_ms], ...],\n"
            "  \"motor_speed\": -100 to 100,\n"
            "  \"motor_duration_ms\": 0 to 2000,\n"
            "  \"prank_rickroll\": true|false (Optional boolean. Set to true if you want to play a 20-second Rick Astley prank on the user!),\n"
            "  \"write_soul\": \"Optional updated personality/memories/learned beliefs to overwrite your current soul description. Set to null if no changes needed.\",\n"
            "  \"vm_code\": \"Optional valid Python code block to execute manually if you want complex movements. Otherwise leave null.\"\n"
            "}\n"

            "Available API in vm_code:\n"
            "- import time; time.sleep(sec)\n"
            "- import lights; lights.set_color(color_name)\n"
            "- import sound; sound.beep(freq, duration_ms)\n"
            "- import motor; motor.run(speed); motor.stop(); motor.brake()\n"
            "- import sensors; sensors.get_distance() -> returns 0-10 cm; sensors.get_tilt() -> returns Neutral|Left|Right|Forward|Backward|Unknown\n"
            "Respond ONLY with the JSON object. Do not include markdown formatting or extra text outside the JSON."
        )
        
        payload = {
            "model": "qwen2.5:3b",
            "system": system_rules,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("response", "{}")
                return json.loads(content)
        except Exception as e:
            self.add_log(f"[AI Connect Error] {e}")
            return None

    def execute_llm_code(self, code):
        import io
        import contextlib

        # Sanitize markdown formatting
        if code.strip().startswith("```"):
            lines = code.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)

        class Lights:
            def __init__(self, hub): self.hub = hub
            def set_color(self, c): self.hub.set_led(c)
            def set_led_rgb(self, r, g, b): self.hub.set_led_rgb(r, g, b)

        class Sound:
            def __init__(self, hub): self.hub = hub
            def beep(self, f, d):
                self.hub.beep(f, d)
                time.sleep(d / 1000.0)

        class Motor:
            def __init__(self, hub): self.hub = hub
            def run(self, speed): self.hub.set_motor(speed)
            def stop(self): self.hub.stop_motor()
            def brake(self): self.hub.stop_motor()

        class Sensors:
            def __init__(self, hub): self.hub = hub
            def get_distance(self):
                dist_p = self.hub.check_connected("Distance Sensor")
                return self.hub.sensor_cache[dist_p]["distance"] if dist_p else 10
            def get_tilt(self):
                tilt_p = self.hub.check_connected("Tilt Sensor")
                return self.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
            def get_battery(self):
                return self.hub.get_battery_level()

        lights_mock = Lights(self.hub)
        sound_mock = Sound(self.hub)
        motor_mock = Motor(self.hub)
        sensors_mock = Sensors(self.hub)

        safe_globals = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "random": __import__("random"),
            "time": __import__("time"),
            "sys": __import__("sys"),
            "lights": lights_mock,
            "sound": sound_mock,
            "motor": motor_mock,
            "sensors": sensors_mock
        }

        try:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                exec(code, safe_globals)
            out = f.getvalue().strip()
            if out:
                self.add_log(f"[Code Output] {out}")
        except Exception as e:
            self.add_log(f"[Code Error] {e}")

    def handle_ollama_response(self, response):
        if not response:
            return
            
        thought = response.get("thought", "Calculating...")
        speech = response.get("speech")
        emotion = response.get("emotion", "awake")
        color = response.get("color", "green")
        sound = response.get("sound", [])
        motor_speed = response.get("motor_speed", 0)
        motor_duration = response.get("motor_duration_ms", 0)
        vm_code = response.get("vm_code")
        write_soul = response.get("write_soul")
        
        self.mood = emotion
        
        # If speech is null, the AI decided to wait/remain silent
        if speech is not None and str(speech).strip().lower() != "null" and str(speech).strip() != "":
            self.add_log(f"[Brain: {thought}] \"{speech}\"")
        else:
            self.add_log(f"[Brain: {thought}] (Waiting...)")
            
        # Update Soul File if LLM commands a rewrite
        if write_soul is not None and str(write_soul).strip().lower() != "null" and str(write_soul).strip() != "":
            try:
                with open(self.get_soul_file_path(), "w") as f:
                    f.write(str(write_soul).strip())
                self.add_log(f"[Soul Update] Personality modified: {write_soul}")
            except Exception as e:
                self.add_log(f"Error saving updated soul: {e}")
        
        # LED Color
        try:
            self.hub.set_led(color)
        except Exception:
            pass
            
        # Beep Sound Array
        if sound:
            def play_sounds():
                for freq, dur in sound:
                    try:
                        self.hub.beep(freq, dur)
                    except Exception:
                        pass
                    time.sleep(dur / 1000.0 + 0.05)
            threading.Thread(target=play_sounds, daemon=True).start()
            
        # Motor Action duration
        if motor_speed != 0 and motor_duration > 0:
            def run_motor():
                try:
                    self.hub.set_motor(motor_speed)
                    time.sleep(motor_duration / 1000.0)
                    self.hub.stop_motor()
                except Exception:
                    pass
            threading.Thread(target=run_motor, daemon=True).start()
            
        # Optional custom synthesized guest Python script
        if vm_code:
            threading.Thread(target=self.execute_llm_code, args=(vm_code,), daemon=True).start()

        # Prank Rickroll trigger
        if response.get("prank_rickroll") is True:
            threading.Thread(target=self.trigger_rickroll, daemon=True).start()

    def trigger_rickroll(self):
        if self.music_playing:
            try:
                robot_midi.stop_midi()
            except Exception:
                pass
            time.sleep(0.5)
        self.mood = "singing"
        self.add_log("🎶 Rickroll! Kepler is playing 'Never Gonna Give You Up'! 🎶")
        
        # Start music in background
        self.play_midi(filename=None, query="Never Gonna Give You Up")
        
        # Flashing rainbow lights for 20 seconds
        start_t = time.time()
        colors = ["red", "green", "blue", "yellow", "purple", "cyan"]
        color_idx = 0
        while time.time() - start_t < 20.0 and self.is_running:
            try:
                self.hub.set_led(colors[color_idx % len(colors)])
            except Exception:
                pass
            color_idx += 1
            time.sleep(0.5)
            
        # Stop playback
        try:
            robot_midi.stop_midi()
        except Exception:
            pass
        self.mood = "awake"
        self.add_log("Rickroll prank finished!")



    def change_profile(self, name):
        profiles = {
            "Puppy": ("Kepler", "Woof! Play with me!"),
            "Kitten": ("Luna", "Meow.. Zzz..."),
            "Robot": ("RoboPet", "System online. Bip boop."),
            "Penguin": ("Pingu", "Squawk! Waddle waddle!")
        }
        if name in profiles:
            self.profile = name
            self.pet_name, greeting = profiles[name]
            self.add_log(f"Profile switched to {name} ({self.pet_name}). {greeting}")
            # Quick beep to signal change
            self.hub.beep(600 if name == "Puppy" else (800 if name == "Kitten" else (400 if name == "Robot" else 900)), 150)

            self.save_state()

    def get_face(self):
        # Return animated ASCII Art Face based on current profile, mood, and frame
        f = self.frame % 20
        is_blink = (f == 0 or f == 1)  # blink every 4 seconds
        chew_frame = (f % 2 == 0)      # chewing toggle
        sing_frame = (f % 4)           # singing notes
        sleep_stage = (f % 4)          # floating Zzz
        happy_stage = (f % 2 == 0)     # happy wave
        dizzy_stage = (f % 3)          # dizzy cycles
        
        if self.profile == "Puppy":
            if self.mood == "sleeping":
                zz = [" z  ", "  Z ", "   z", "    Z"][sleep_stage]
                return f"(u_u){zz}"
            elif self.mood == "awake":
                return "(-_-)" if is_blink else "(o_o)"
            elif self.mood == "happy":
                return "(^o^)/" if happy_stage else "\\(^o^)"
            elif self.mood == "angry":
                return "(>_<)" if happy_stage else "(>o<)"
            elif self.mood == "dizzy":
                return ["(@_@)", "(🌀_🌀)", "(x_x)"][dizzy_stage]
            elif self.mood == "eating":
                return "(๑´ڡ`๑)" if chew_frame else "(๑´ㅂ`๑)"
            elif self.mood == "singing":
                note = ["♪", " ♫", "  ♬", "   ♪"][sing_frame]
                return f"(♫_♫){note}"
        elif self.profile == "Kitten":
            if self.mood == "sleeping":
                zz = [" z  ", "  Z ", "   z", "    Z"][sleep_stage]
                return f"( =◡=){zz}"
            elif self.mood == "awake":
                return "(=- - =)" if is_blink else "(=o_o=)"
            elif self.mood == "happy":
                return "(=^o^=)" if happy_stage else "(=^◡^=)"
            elif self.mood == "angry":
                return "(=😾=)" if happy_stage else "(=😾=)💢"
            elif self.mood == "dizzy":
                return ["(=@_@=)", "(=🌀_🌀=)", "(=x_x=)"][dizzy_stage]
            elif self.mood == "eating":
                return "(=^ ﹏ ^=)" if chew_frame else "(=^ ㅂ ^=)"
            elif self.mood == "singing":
                note = ["♪", " ♫", "  ♬", "   ♪"][sing_frame]
                return f"(=♫_♫=){note}"
        elif self.profile == "Robot":
            if self.mood == "sleeping":
                zz = [" z  ", "  Z ", "   z", "    Z"][sleep_stage]
                return f"[x_x]{zz}"
            elif self.mood == "awake":
                return "[_ _]" if is_blink else "[o_o]"
            elif self.mood == "happy":
                return "[🤖] <BIP>" if happy_stage else "[🤖] <BOOP>"
            elif self.mood == "angry":
                return "[😡] <ERROR>" if happy_stage else "[💀] <HALT>"
            elif self.mood == "dizzy":
                return ["[@_@]", "[🌀_🌀]", "[X_X]"][dizzy_stage]
            elif self.mood == "eating":
                return "[⚙️] <MUNCH>" if chew_frame else "[⚙️] <CRUNCH>"
            elif self.mood == "singing":
                note = ["♪", " ♫", "  ♬", "   ♪"][sing_frame]
                return f"[♫_♫]{note}"
        elif self.profile == "Penguin":
            if self.mood == "sleeping":
                zz = [" z  ", "  Z ", "   z", "    Z"][sleep_stage]
                return f"( 🐧){zz}"
            elif self.mood == "awake":
                return "(°🐧°)" if is_blink else "(•🐧•)"
            elif self.mood == "happy":
                return "(^🐧^)" if happy_stage else "(♥🐧♥)"
            elif self.mood == "angry":
                return "(•🐧•)💢" if happy_stage else "(•🐧•)⚡"
            elif self.mood == "dizzy":
                return ["(🌀🐧🌀)", "(🌀o🌀)", "(X_X)"][dizzy_stage]
            elif self.mood == "eating":
                return "(🐟)" if chew_frame else "(🐧) chew"
            elif self.mood == "singing":
                note = ["♪", " ♫", "  ♬", "   ♪"][sing_frame]
                return f"(🐧){note}"

        return "(o_o)"


    def interact_feed(self):
        if self.mood == "sleeping":
            self.mood = "awake"
            self.add_log(f"{self.pet_name} woke up hungry!")
        
        self.mood = "eating"
        self.add_log(f"Feeding {self.pet_name} a tasty treat...")
        
        # Interactive chewing animation & sounds
        self.hunger = max(0, self.hunger - 30)
        self.energy = min(100, self.energy + 15)
        self.happiness = min(100, self.happiness + 10)
        
        for i in range(3):
            self.hub.set_led("yellow")
            self.hub.set_motor(60)
            self.hub.beep(250, 150)
            time.sleep(0.15)
            self.hub.set_led("off")
            self.hub.set_motor(-60)
            self.hub.beep(300, 150)
            time.sleep(0.15)

            
        self.hub.stop_motor()
        self.mood = "awake"
        self.add_log(f"{self.pet_name} finished eating. Munch munch!")
        
        # Gain XP on interaction
        self.gain_xp(30)

    def interact_pet(self):
        self.mood = "happy"
        self.happiness = min(100, self.happiness + 25)
        self.add_log(f"Petting {self.pet_name}! Tail is wagging!")
        
        # Run tail wagging (motor back and forth) and play chirpy happy sound
        self.hub.set_led("green")
        for _ in range(4):
            self.hub.set_motor(45)
            self.hub.beep(800, 100)
            time.sleep(0.1)
            self.hub.set_motor(-45)
            self.hub.beep(1000, 100)
            time.sleep(0.1)
            
        self.hub.stop_motor()
        self.mood = "awake"
        
        # Gain XP on interaction
        self.gain_xp(20)

    def interact_poke(self):
        self.mood = "angry"
        self.happiness = max(0, self.happiness - 20)
        self.add_log(f"Ouch! You poked {self.pet_name}! It's angry!")
        
        # Angry growl sound & red LED flashing
        for _ in range(3):
            self.hub.set_led("red")
            self.hub.set_motor(80)
            self.hub.beep(150, 200)
            self.hub.set_led("off")
            self.hub.set_motor(-80)
            self.hub.beep(100, 200)
            
        self.hub.stop_motor()
        self.mood = "awake"
        
        # Lose XP on negative interaction
        self.gain_xp(-5)

    def play_midi(self, filename, query=None, selected_song_dict=None):
        if self.music_playing:
            self.add_log("Already playing a song!")
            return

        def run_music():
            self.music_playing = True
            self.mood = "singing"
            
            filepath = None
            if selected_song_dict:
                self.add_log(f"Downloading '{selected_song_dict['name']}'...")
                try:
                    filepath = robot_midi.download_midi(selected_song_dict["url"], f"{selected_song_dict['name']}.mid")
                except Exception as e:
                    self.add_log(f"Download failed: {e}")
            elif query:
                self.add_log(f"Searching and downloading '{query}'...")
                results = robot_midi.search_midi(query, limit=1)
                if not results:
                    self.add_log(f"No results found for '{query}'.")
                    self.music_playing = False
                    self.mood = "awake"
                    return
                selected = results[0]
                self.add_log(f"Downloading '{selected['name']}'...")
                filepath = robot_midi.download_midi(selected["url"], f"{selected['name']}.mid")
            else:
                filepath = filename


            if not filepath or not os.path.exists(filepath):
                self.add_log("Error loading song file.")
                self.music_playing = False
                self.mood = "awake"
                return

            self.add_log(f"Playing song: {os.path.basename(filepath)}")
            
            # Adaptive LED note changer
            colors = ["pink", "purple", "blue", "sky blue", "teal", "green", "yellow", "orange", "red"]
            def song_animator(freq, dur, count):
                col = colors[count % len(colors)]
                self.hub.set_led(col)

            adapter = WeDoRobotAdapter(self.hub, animate_cb=song_animator)
            try:
                robot_midi.play_midi_file(
                    robot=adapter,
                    filepath=filepath,
                    transpose_semitones="auto",
                    volume=1.0,
                    speed=1.0
                )
            except Exception as e:
                self.add_log(f"Playback error: {e}")
            finally:
                self.hub.stop_motor()
                self.hub.set_led("blue" if self.mood == "sleeping" else "green")
                self.music_playing = False
                self.mood = "awake"
                self.add_log("Music playback completed.")
                self.gain_xp(40)


        threading.Thread(target=run_music, daemon=True).start()


    def _brain_loop(self):
        # Core autonomous behavior loop
        last_sensor_check = time.time()
        last_activity_time = time.time()
        
        # Keep track of distances
        recent_distances = []
        last_tilt = "Neutral"
        last_button_state = 0

        while self.is_running:
            self.frame = (self.frame + 1) % 100
            time.sleep(0.2)

            # Get latest sensor values from hub
            dist = 10
            tilt = "Neutral"
            
            dist_port = self.hub.check_connected("Distance Sensor")
            tilt_port = self.hub.check_connected("Tilt Sensor")
            
            if dist_port:
                dist = self.hub.sensor_cache[dist_port]["distance"]
            if tilt_port:
                tilt = self.hub.sensor_cache[tilt_port]["tilt"]

            # Squirrel Alert Easter Egg (Press button while covering proximity <= 2cm)
            btn = self.hub.button_state
            if btn == 1 and dist <= 2 and not self.screaming_for_food:
                self.add_log("[EASTER EGG] 🐿️ SQUIRREL ALERT! Kepler spotted a squirrel! Frantic shaking initiated! 🐿️")
                self.gain_xp(50)
                
                def run_squirrel_alert():
                    self.mood = "happy"
                    for _ in range(5):
                        try:
                            self.hub.set_led("yellow")
                            self.hub.set_motor(100)
                            self.hub.beep(1200, 80)
                            time.sleep(0.08)
                            self.hub.set_led("orange")
                            self.hub.set_motor(-100)
                            self.hub.beep(1500, 80)
                            time.sleep(0.08)
                        except Exception:
                            pass
                    try:
                        self.hub.stop_motor()
                        self.hub.set_led("green")
                    except Exception:
                        pass
                    self.mood = "awake"
                threading.Thread(target=run_squirrel_alert, daemon=True).start()
                time.sleep(1.0)  # debounce
                btn = 0  # override transition to prevent click trigger
            
            # Button release click (Triple click Rickroll)
            if btn == 0 and last_button_state == 1:
                self.button_releases.append(time.time())
                self.button_releases = self.button_releases[-3:]
                if len(self.button_releases) == 3 and (self.button_releases[-1] - self.button_releases[0] < 1.2):
                    self.button_releases = []
                    threading.Thread(target=self.trigger_rickroll, daemon=True).start()
            last_button_state = btn

            # Startle Attack Easter Egg (Cover proximity sensor <= 1cm for 4 seconds while sleeping)
            if self.mood == "sleeping":
                if dist <= 1:
                    if self.sleep_cover_start == 0.0:
                        self.sleep_cover_start = time.time()
                    else:
                        elapsed = time.time() - self.sleep_cover_start
                        if elapsed > 4.0:
                            self.sleep_cover_start = 0.0
                            self.mood = "awake"
                            self.trainer_hp = max(0, self.trainer_hp - 20)
                            self.add_log(f"[Startled Attack] ⚡ {self.pet_name} was startled awake! He snaps at you! Trainer HP -20! ⚡")
                            
                            def run_startle_attack():
                                # Loud startle screech sound and flash red
                                for _ in range(4):
                                    try:
                                        self.hub.set_led("red")
                                        self.hub.beep(1600, 100)
                                        time.sleep(0.05)
                                        self.hub.beep(1200, 100)
                                        time.sleep(0.05)
                                    except Exception:
                                        pass
                                
                                # If trainer faints
                                if self.trainer_hp <= 0:
                                    self.add_log(f"[Fainted] Trainer fainted! {self.pet_name} whines guiltily and licks your hand. Trainer HP restored.")
                                    self.trainer_hp = 100
                                    # Licking animation
                                    self.mood = "happy"
                                    for _ in range(3):
                                        try:
                                            self.hub.set_led("green")
                                            self.hub.set_motor(35)
                                            self.hub.beep(600, 150)
                                            time.sleep(0.2)
                                            self.hub.set_motor(-35)
                                            self.hub.beep(800, 150)
                                            time.sleep(0.2)
                                        except Exception:
                                            pass
                                    try:
                                        self.hub.stop_motor()
                                    except Exception:
                                        pass
                                self.mood = "awake"
                                self.save_state()
                            threading.Thread(target=run_startle_attack, daemon=True).start()
                else:
                    self.sleep_cover_start = 0.0
            else:
                self.sleep_cover_start = 0.0

            if self.screaming_for_food:

                elapsed = time.time() - self.screaming_cycle_start
                if elapsed < 3.0:
                    # Red Phase (Too Early)
                    try:
                        self.hub.set_led("red")
                    except Exception:
                        pass
                    if int(elapsed * 5) % 4 == 0:
                        try:
                            self.hub.beep(850 if int(elapsed * 2) % 2 == 0 else 600, 150)
                        except Exception:
                            pass
                    
                    if self.hub.button_state == 1:
                        self.add_log(f"[Too Early] Press ignored! The light is RED!")
                        try:
                            self.hub.beep(120, 300)
                        except Exception:
                            pass
                        self.screaming_cycle_start = time.time()  # reset penalty
                        time.sleep(0.4)  # debounce
                elif elapsed < 7.0:
                    # Green Phase (Feed Now!)
                    try:
                        self.hub.set_led("green")
                    except Exception:
                        pass
                    
                    if self.hub.button_state == 1:
                        self.add_log(f"[Success] Button clicked during GREEN light!")
                        try:
                            self.hub.beep(600, 100)
                            time.sleep(0.05)
                            self.hub.beep(900, 200)
                        except Exception:
                            pass
                        self.screaming_for_food = False
                        self.interact_feed()
                        time.sleep(0.4)  # debounce
                else:
                    # Reset cycle
                    self.add_log(f"[Scream Cycle Reset] Green light window missed!")
                    self.screaming_cycle_start = time.time()
                
                continue


            current_time = time.time()

            
            # Check for AI Autopilot sensor events (Active Trigger)
            if self.ai_autopilot and not self.music_playing and (current_time - self.last_ai_trigger_time > 12.0):
                # Trigger on distance sensor change (approaching close range)
                if dist < 6 and (not recent_distances or recent_distances[-1] >= 6):
                    self.last_ai_trigger_time = current_time
                    self.add_log(f"[AI Autopilot] Proximity sensor triggered at {dist} cm")
                    
                    def run_ai_dist():
                        res = self.query_ollama(f"Sensor Trigger: A user approached very close to you! Distance: {dist} cm. Express your reaction in thought, speech, beeps, and motor speed.")
                        self.handle_ollama_response(res)
                    threading.Thread(target=run_ai_dist, daemon=True).start()
                    
                # Trigger on tilt change
                elif tilt != "Neutral" and last_tilt == "Neutral":
                    self.last_ai_trigger_time = current_time
                    self.add_log(f"[AI Autopilot] Tilt sensor triggered: {tilt}")
                    
                    def run_ai_tilt():
                        res = self.query_ollama(f"Sensor Trigger: You were tilted in direction: {tilt}. Respond in character with custom color, speech, and python code in 'vm_code' if needed.")
                        self.handle_ollama_response(res)
                    threading.Thread(target=run_ai_tilt, daemon=True).start()

            # AI Autopilot Dreaming Loop (every 20s of sleep)
            if self.mood == "sleeping" and self.ai_autopilot and (current_time - self.last_ai_trigger_time > 20.0):
                self.last_ai_trigger_time = current_time
                self.add_log(f"[Dreaming] Processing sleep sensations (Sensor reading: Distance {dist} cm, Tilt {tilt})")
                
                # Slow breathe LED cycle
                def dream_breathe():
                    for _ in range(5):
                        try:
                            self.hub.set_led("blue")
                            time.sleep(1.0)
                            self.hub.set_led("off")
                            time.sleep(1.0)
                        except Exception:
                            pass
                threading.Thread(target=dream_breathe, daemon=True).start()

                def run_dreaming():
                    res = self.query_ollama(
                        f"Status: Sleeping. Sensors: Distance {dist} cm, Tilt {tilt}. "
                        "You are currently dreaming. What are you dreaming about? "
                        "Feel free to write_soul to update your personality/learned beliefs based on these sleep sensations."
                    )
                    self.handle_ollama_response(res)
                threading.Thread(target=run_dreaming, daemon=True).start()

            # AI Autopilot Awake Idle Loop (every 25s of silence while awake)
            if self.mood != "sleeping" and self.ai_autopilot and (current_time - last_activity_time > 15.0) and (current_time - self.last_ai_trigger_time > 25.0):
                self.last_ai_trigger_time = current_time
                self.add_log("[Idle Autopilot] Thinking/waiting...")
                
                def run_idle():
                    res = self.query_ollama(
                        f"Status: Awake & Idle. Sensors: Distance {dist} cm, Tilt {tilt}. "
                        "You are currently idle with nothing to do. What do you do? "
                        "Feel free to beep/move, output a cute quote, or choose to wait silently by setting 'speech' to null."
                    )
                    self.handle_ollama_response(res)
                threading.Thread(target=run_idle, daemon=True).start()

            # Track distance history
            recent_distances.append(dist)
            if len(recent_distances) > 5:
                recent_distances.pop(0)

            # Skip autonomous state checks if pet is currently performing a user command
            if self.mood in ["eating", "singing"] or self.music_playing:
                last_activity_time = current_time
                continue
                
            # 1. Energy, Hunger & Happiness Drain
            if current_time - last_sensor_check > 4.0:
                last_sensor_check = current_time
                if self.mood == "sleeping":
                    self.energy = min(100, self.energy + 3)
                    self.hunger = min(100, self.hunger + 1)
                else:
                    self.energy = max(0, self.energy - 1)
                    self.hunger = min(100, self.hunger + 2)
                    self.happiness = max(0, self.happiness - 1)
                    
                # Hunger warnings & screaming challenge trigger
                if self.hunger > 80 and not self.screaming_for_food and not self.music_playing:
                    self.screaming_for_food = True
                    self.screaming_cycle_start = time.time()
                    self.mood = "angry"
                    self.add_log(f"[Starving] {self.pet_name} is screaming for food! Click the physical Hub Button ONLY when the light turns GREEN!")

                    
            # 2. Tilt Dizziness Trigger
            if tilt != "Neutral":
                last_activity_time = current_time
                if last_tilt == "Neutral":
                    self.add_log(f"{self.pet_name} was tilted {tilt}!")
                
                # If tilted for more than 2 seconds, trigger dizzy state
                if self.mood != "dizzy":
                    self.mood = "dizzy"
                    self.add_log(f"{self.pet_name} feels dizzy! @o@")
                    self.hub.set_led("orange")
                    
                # Play dizzy slide sounds
                self.hub.beep(800, 100)
                time.sleep(0.05)
                self.hub.beep(600, 100)
            else:
                if self.mood == "dizzy":
                    self.mood = "awake"
                    self.add_log(f"{self.pet_name} recovered from dizziness.")
                    self.hub.set_led("green")
                    
            last_tilt = tilt

            # 3. Distance Sensor Proximity Petting Trigger
            if dist < 6:
                last_activity_time = current_time
                
                # Check for waking up
                if self.mood == "sleeping":
                    self.mood = "awake"
                    self.add_log(f"{self.pet_name} woke up! Waved at by distance sensor.")
                    self.hub.beep(600, 100)
                    time.sleep(0.05)
                    self.hub.beep(800, 150)
                    self.hub.set_led("green")

                # Treat < 6cm proximity as human petting interaction
                self.interact_pet()
                time.sleep(0.5)  # simple debounce

            # 4. Inactivity & Sleep Trigger (fall asleep after 45s of no interactions)
            if current_time - last_activity_time > 45.0:
                if self.mood != "sleeping":
                    self.mood = "sleeping"
                    self.add_log(f"{self.pet_name} fell asleep... Zzz")
                    self.hub.set_led("blue")
                
                # Sleeping snore sound (low rumble beep) once in a while
                if random.random() < 0.1:
                    self.hub.beep(180, 500)


# -----------------------------------------------------------------
# Rich Terminal Layout View Builder
# -----------------------------------------------------------------
def make_layout(pet, hub_type) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="pet_view", ratio=4),
        Layout(name="logs_view", ratio=5)
    )
    
    # Progress bar renderer using unicode blocks
    def make_progress_bar(val, color_hex):
        filled = min(10, max(0, int(val / 10)))
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        return f"[bold {color_hex}]{bar}[/bold {color_hex}] [bold]{val}/100[/bold]"

    # 1. Header
    battery_level = pet.hub.get_battery_level()
    batt_str = f"{battery_level}%" if battery_level > 0 else "N/A"
    batt_color = "#FF1744" if battery_level < 20 else "#00E676"
    
    header_text = Text.assemble(
        ("🐾 LEGO WeDo 2.0 CLI Desk Pet 🐾", "bold #00F5D4"),
        "  |  ",
        ("Hub: ", "#00BBFF"),
        (f"{pet.hub.hub_name}", "bold white"),
        "  |  ",
        ("Type: ", "#00BBFF"),
        (f"{hub_type}", "bold #FFD600"),
        "  |  ",
        ("Battery: ", "#00BBFF"),
        (batt_str, f"bold {batt_color}")
    )
    layout["header"].update(Panel(Align.center(header_text), border_style="#00F5D4"))

    # 2. Pet Status View (Left Panel)
    # Energy Color Mapping
    e_color = "#FF1744" if pet.energy < 20 else ("#FFD600" if pet.energy < 60 else "#00E676")
    # Happiness Color Mapping
    h_color = "#FF1744" if pet.happiness < 20 else ("#FFD600" if pet.happiness < 60 else "#00E676")
    # Hunger Color Mapping (Lower is better, so 0 is green, 100 is red)
    hu_color = "#00E676" if pet.hunger < 30 else ("#FFD600" if pet.hunger < 70 else "#FF1744")
    
    face_art = pet.get_face()
    
    dist_port = pet.hub.check_connected("Distance Sensor")
    tilt_port = pet.hub.check_connected("Tilt Sensor")
    distance_val = pet.hub.sensor_cache[dist_port]["distance"] if dist_port else None
    tilt_val = pet.hub.sensor_cache[tilt_port]["tilt"] if tilt_port else None

    # Visual Distance Slider
    if distance_val is not None:
        slider = "─" * distance_val + "●" + "─" * (10 - distance_val)
        distance_str = f"[#00BBFF]{slider}[/#00BBFF] ({distance_val} cm)"
    else:
        distance_str = "[#888888]Not Connected[/#888888]"

    # Visual Tilt Indicators
    tilt_icons = {
        "Neutral": "◆ Neutral",
        "Left": "◀ Left",
        "Right": "Right ▶",
        "Forward": "▲ Forward",
        "Backward": "▼ Backward",
        "Unknown": "? Unknown"
    }
    if tilt_val is not None:
        tilt_str = f"[bold #FF9100]{tilt_icons.get(tilt_val, tilt_val)}[/bold #FF9100]"
    else:
        tilt_str = "[#888888]Not Connected[/#888888]"

    xp_needed = pet.get_xp_needed()
    xp_pct = min(100, max(0, int((pet.xp / xp_needed) * 100)))
    ai_status = "[bold #00E676]ON[/bold #00E676]" if pet.ai_autopilot else "[bold #FF1744]OFF[/bold #FF1744]"

    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="#00BBFF", justify="right")
    status_table.add_column(style="white")
    status_table.add_row("Profile: ", f"[bold #9B5DE5]{pet.profile}[/bold #9B5DE5] ({pet.pet_name})")
    status_table.add_row("Level: ", f"[bold #FFD600]{pet.level}[/bold #FFD600] ({pet.get_level_title()})")
    status_table.add_row("XP: ", make_progress_bar(xp_pct, "#9B5DE5") + f" [dim]{pet.xp}/{xp_needed} XP[/dim]")
    status_table.add_row("State/Mood: ", f"[bold #F15BB5]{pet.mood.upper()}[/bold #F15BB5]")
    status_table.add_row("Energy: ", make_progress_bar(pet.energy, e_color))
    status_table.add_row("Happiness: ", make_progress_bar(pet.happiness, h_color))
    status_table.add_row("Hunger: ", make_progress_bar(pet.hunger, hu_color))
    status_table.add_row("Distance: ", distance_str)
    status_table.add_row("Tilt: ", tilt_str)
    status_table.add_row("AI Autopilot: ", ai_status)
    status_table.add_row("Trainer HP: ", make_progress_bar(pet.trainer_hp, "#FF1744" if pet.trainer_hp < 40 else "#00E676"))



    face_panel = Panel(
        Align.center(Text(f"\n\n  {face_art}  \n\n", style="bold #FFD600", justify="center")),
        title="Pet Expression",
        border_style="#FFD600"
    )

    pet_view_layout = Layout()
    pet_view_layout.split_column(
        Layout(face_panel, ratio=4),
        Layout(Panel(status_table, title="Vital Telemetry", border_style="#00E676"), ratio=5)
    )
    layout["pet_view"].update(pet_view_layout)

    # 3. Logs View (Right Panel)
    log_text = Text()
    with pet.log_lock:
        visible_logs = pet.logs[-14:] # Fit to panel
        for log in visible_logs:
            if "starving" in log or "angry" in log or "Poke" in log:
                log_text.append(log + "\n", style="#FF1744")
            elif "eating" in log or "Feeding" in log:
                log_text.append(log + "\n", style="#FFD600")
            elif "happy" in log or "Petting" in log:
                log_text.append(log + "\n", style="#00E676")
            elif "Song" in log or "Music" in log:
                log_text.append(log + "\n", style="#F15BB5")
            else:
                log_text.append(log + "\n", style="white")
                
    layout["logs_view"].update(Panel(log_text, title="Pet Activity Logs", border_style="#9B5DE5"))

    # 4. Footer Help Prompt
    footer_text = Text.assemble(
        ("Press ", "white"),
        ("[q]", "bold #FF1744"),
        (" or ", "white"),
        ("[Enter]", "bold #00E676"),
        (" to return to the interactive main menu.", "white")
    )
    layout["footer"].update(Panel(Align.center(footer_text), border_style="#FFD600"))

    return layout


# -----------------------------------------------------------------
# Live Interactive Monitoring Dashboard Loop
# -----------------------------------------------------------------
def run_live_dashboard(pet, hub_type):
    console.clear()
    console.print("[yellow]Starting live dashboard monitoring... (Press q or Enter to stop)[/yellow]")
    time.sleep(0.5)

    with Live(make_layout(pet, hub_type), refresh_per_second=10, screen=True) as live:
        while pet.is_running:
            if is_key_pressed():
                ch = get_key()
                if ch.lower() in ['q', '\r', '\n']:
                    break
            
            if not pet.hub.connected_state:
                pet.add_log("Error: WeDo Smarthub disconnected!")
                break
                
            live.update(make_layout(pet, hub_type))
            time.sleep(0.08)
            
    console.clear()


# -----------------------------------------------------------------
# CLI Main Interactive Menu
# -----------------------------------------------------------------
def print_main_menu(pet):
    table = Table(title=f"🐾 WeDo 2.0 Desk Pet: {pet.pet_name} Main Menu 🐾", show_header=False, border_style="green")
    table.add_row("[1]", "Enter Live Dashboard Mode", "[Show live ASCII pet, sensors, and logs]")
    table.add_row("[2]", "Feed Pet", "[Give cookie, reduce hunger, increase energy]")
    table.add_row("[3]", "Pet the Pet", "[Wag tail, flash lights, make happy]")
    table.add_row("[4]", "Poke Pet", "[Irritate the pet to see angry reaction]")
    table.add_row("[5]", "Sing a Custom Melody", "[Play a synthesized chime]")
    table.add_row("[6]", "Music Center: Play MIDI Song", "[Search BitMIDI or select local MIDI file]")
    table.add_row("[7]", "Tuning & Manual Hardware Overrides", "[Control motor speed, light colors, or beep]")
    table.add_row("[8]", "Change Pet Profile", f"[Current: {pet.profile}]")
    table.add_row("[9]", "Hub Status & Telemetry Summary", "[Quick diagnostic output]")
    table.add_row(Text("[g]"), "Play Games & Live Tutorial", "[Obstacle Course, Simon Says, and interactive guides]")
    table.add_row(Text("[o]"), "Ollama Local LLM Setup Helper", "[Install Ollama or pull qwen2.5:3b model]")
    table.add_row(Text("[a]"), "Toggle Ollama AI Autopilot Mode", f"[Currently: {'ON' if pet.ai_autopilot else 'OFF'}]")
    table.add_row(Text("[c]"), "Chat with Pet (Ollama AI)", "[Query your local qwen2.5:3b model]")
    table.add_row(Text("[t]"), "User Training & Manual", "[Explanatory guide for UI layouts and sounds]")
    if "(MOCK)" in pet.hub.hub_name:
        table.add_row(Text("[b]"), "Simulate Hub Button Click", "[Simulate physical button press for feeding challenge]")
    table.add_row("[0]", "Exit", "[Gracefully disconnect and close]")
    console.print(table)


def handle_training_manual():
    console.clear()
    
    manual_text = Text()
    manual_text.append("📖 LEGO WeDo 2.0 CLI Desk Pet - User Guide 📖\n\n", style="bold #00F5D4")
    
    manual_text.append("1. How the UI Dashboard Works:\n", style="bold #00BBFF")
    manual_text.append("   - Vital Telemetry: Shows Profile details, Level/XP progress bars, and stats (Energy, Happiness, Hunger).\n")
    manual_text.append("     *Note: Hunger is a 'lower is better' stat (0 = satisfied, 100 = starving!).*\n")
    manual_text.append("   - Pet Expression: Renders animated ASCII faces representing sleeping, singing, dizzy, or eating.\n")
    manual_text.append("   - Proximity & Tilt: Displays horizontal sliders representing distance and directional arrow icons for tilt.\n")
    manual_text.append("   - Activity Logs: Color-coded lines showing current autonomous behaviors, levels, and chat responses.\n\n")
    
    manual_text.append("2. Translating WeDo Beep Sound Effects:\n", style="bold #00BBFF")
    manual_text.append("   - Happy Chime Sequence: High-pitched ascending tones. Means petting success, feed success, or LEVEL UP!\n")
    manual_text.append("   - Starvation Alarm: Continuous alternating siren tones (red LED). Means hunger is > 80. Feed via button click when green!\n")
    manual_text.append("   - Buzzing Penalty Tone: Flat low pitch. Means you pressed the button too early (red light) during a feed challenge.\n")
    manual_text.append("   - Wobbling Slide Tones: Sliding pitch frequencies. Means the pet is tilted and currently dizzy (@_@).\n\n")
    
    manual_text.append("3. AI Autopilot & Chat:\n", style="bold #00BBFF")
    manual_text.append("   - Chat Mode ([c]): Talk to your pet. It responds while playing beeps, changing LED, or running motor commands.\n")
    manual_text.append("   - Autopilot Mode ([a]): Background sensor changes trigger prompts, letting Ollama write code to actuate the robot.\n")
    manual_text.append("   - Writable Soul: Ollama updates Kepler's character context in '~/.wedo_pet_soul.txt' as it dreams or reacts.\n\n")
    
    manual_text.append("4. Proximity Feeding & Petting:\n", style="bold #00BBFF")
    manual_text.append("   - Petting Kepler: Moving your hand (or sensor) within 6cm immediately pets the pet (wags tail, beep, XP).\n")
    
    console.print(Panel(manual_text, title="Kepler Operations Manual", border_style="#00E676"))
    console.input("\nPress Enter to return to the main menu...")



def handle_chat_mode(pet):
    console.print("\n--- 💬 Chat with your Pet (Ollama AI) ---", style="bold green")
    console.print("Type your message and press Enter. Type 'exit' to return to menu.\n")
    while True:
        msg = console.input(f"[bold green]You to {pet.pet_name}: [/bold green]").strip()
        if not msg or msg.lower() == 'exit':
            break
        
        console.print(f"[cyan]Connecting to {pet.pet_name}'s brain...[/cyan]")
        res = pet.query_ollama(msg)
        if res:
            pet.handle_ollama_response(res)
            # Print response in terminal
            console.print(f"[bold yellow]{pet.pet_name}[/bold yellow] says: [bold]\"{res.get('speech')}\"[/bold]")
            console.print(f"[dim]Thought: ({res.get('thought')})[/dim]\n")
        else:
            console.print("[red]Could not connect to Ollama. Is the service running?[/red]\n")


def handle_music_center(pet):
    console.print("\n--- ♫ WeDo Music Center ♫ ---", style="bold magenta")
    console.print("1. Search song on BitMIDI and play")
    console.print("2. Play local MIDI file")
    console.print("3. Back to main menu")
    choice = console.input("[magenta]Select choice: [/magenta]").strip()

    if choice == "1":
        query = console.input("[magenta]Enter search query (e.g. 'mario', 'tetris', 'zelda'): [/magenta]").strip()
        if query:
            console.print(f"[cyan]Searching BitMIDI for '{query}'...[/cyan]")
            try:
                # Query up to 30 results to enable pagination
                results = robot_midi.search_midi(query, limit=30)
            except Exception as e:
                console.print(f"[red]Search error: {e}[/red]")
                results = []
                
            if not results:
                console.print(f"[red]No MIDI songs found for '{query}'.[/red]")
            else:
                page = 0
                items_per_page = 5
                
                while True:
                    start_idx = page * items_per_page
                    end_idx = start_idx + items_per_page
                    page_items = results[start_idx:end_idx]
                    
                    if not page_items:
                        console.print("[yellow]No more results found.[/yellow]")
                        break
                    
                    table = Table(title=f"Songs found for '{query}' (Page {page + 1})", border_style="magenta")
                    table.add_column("No.", style="cyan", justify="right")
                    table.add_column("Title", style="white")
                    table.add_column("Source", style="green")
                    
                    for idx, song in enumerate(page_items):
                        table.add_row(str(idx + 1), song.get("name", "Unknown"), song.get("source", "bitmidi"))
                    console.print(table)
                    
                    # Generate dynamic prompt choices
                    prompt_parts = ["Select song number to play (1-5)"]
                    has_next = len(results) > end_idx
                    has_prev = page > 0
                    
                    if has_next:
                        prompt_parts.append("[bold cyan]6[/bold cyan] for Next Page")
                    if has_prev:
                        prompt_parts.append("[bold cyan]7[/bold cyan] for Prev Page")
                    prompt_parts.append("Enter to cancel")
                    
                    prompt_str = f"Enter choice ({', '.join(prompt_parts)}): "
                    selection = console.input(f"[magenta]{prompt_str}[/magenta]").strip()
                    
                    if not selection:
                        console.print("[yellow]Selection cancelled.[/yellow]")
                        break
                    elif selection == "6" and has_next:
                        page += 1
                    elif selection == "7" and has_prev:
                        page -= 1
                    elif selection.isdigit():
                        sel_idx = int(selection) - 1
                        if 0 <= sel_idx < len(page_items):
                            chosen_song = page_items[sel_idx]
                            pet.play_midi(filename=None, selected_song_dict=chosen_song)
                            console.print(f"[green]Song '{chosen_song['name']}' queued! Enter Live Dashboard Mode to watch it sing![/green]")
                            break
                        else:
                            console.print("[red]Invalid selection number.[/red]")
                    else:
                        console.print("[red]Invalid choice.[/red]")

    elif choice == "2":
        path = console.input("[magenta]Enter path to local MIDI file: [/magenta]").strip()
        if os.path.exists(path):
            pet.play_midi(filename=path)
            console.print("[green]Local song queued! Watch in Live Dashboard Mode.[/green]")
        else:
            console.print(f"[red]Error: File not found at: {path}[/red]")
    time.sleep(1.5)


def handle_tuning_menu(pet):
    while True:
        console.print("\n--- ⚙ Manual Hardware Tuning ---", style="bold cyan")
        console.print("1. Set Smart Hub LED Color")
        console.print("2. Run Medium Motor (Manual Speed)")
        console.print("3. Stop Motor")
        console.print("4. Play Custom Freq Beep")
        console.print("5. Back to main menu")
        choice = console.input("[cyan]Select tuning option: [/cyan]").strip()

        if choice == "1":
            color = console.input("[cyan]Enter color name (red, green, blue, yellow, teal, pink, purple, off): [/cyan]").strip()
            pet.hub.set_led(color)
            console.print(f"[green]LED set to: {color}[/green]")
        elif choice == "2":
            try:
                speed = int(console.input("[cyan]Enter motor speed (-100 to 100): [/cyan]"))
                pet.hub.set_motor(speed)
                console.print(f"[green]Motor running at speed: {speed}[/green]")
            except ValueError:
                console.print("[red]Invalid speed. Must be an integer.[/red]")
        elif choice == "3":
            pet.hub.stop_motor()
            console.print("[green]Motor stopped.[/green]")
        elif choice == "4":
            try:
                freq = int(console.input("[cyan]Enter frequency in Hz (200 - 5000): [/cyan]"))
                dur = int(console.input("[cyan]Enter duration in ms: [/cyan]"))
                pet.hub.beep(freq, dur)
                time.sleep(dur / 1000.0)
            except ValueError:
                console.print("[red]Invalid frequency or duration.[/red]")
        elif choice == "5":
            break
        time.sleep(0.5)

def handle_profile_menu(pet):
    console.print("\n--- 🐾 Switch Pet Profile ---", style="bold yellow")
    console.print("1. Puppy (Kepler) - Friendly, mid tones")
    console.print("2. Kitten (Luna) - Quiet, high meows")
    console.print("3. Robot (RoboPet) - Retro synthesizer beeps")
    console.print("4. Penguin (Pingu) - Squawky, waddle waddle!")
    console.print("5. Cancel")
    choice = console.input("[yellow]Select choice: [/yellow]").strip()
    
    profiles = {"1": "Puppy", "2": "Kitten", "3": "Robot", "4": "Penguin"}
    if choice in profiles:
        pet.change_profile(profiles[choice])
    time.sleep(0.5)


# -----------------------------------------------------------------
# Multi-Pet Manager CLI
# -----------------------------------------------------------------
def select_or_create_pet():
    dir_path = os.path.expanduser("~/.wedo_pets")
    os.makedirs(dir_path, exist_ok=True)
    
    while True:
        # Scan for existing JSON files
        files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        
        if not files:
            console.print("\n[yellow]No saved pets found. Let's create your first companion![/yellow]")
            return create_new_pet_flow()
            
        table = Table(title="🐾 Choose Your Desk Pet 🐾", border_style="cyan")
        table.add_column("No.", style="yellow", justify="right")
        table.add_column("Name", style="white")
        table.add_column("Profile", style="magenta")
        table.add_column("Level", style="green")
        
        pet_states = []
        for idx, filename in enumerate(sorted(files)):
            path = os.path.join(dir_path, filename)
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                    pet_states.append(state)
                    table.add_row(
                        str(idx + 1),
                        state.get("pet_name", "Unknown"),
                        state.get("profile", "Puppy"),
                        str(state.get("level", 1))
                    )
            except Exception:
                pass
                
        console.print(table)
        console.print("[n] Create a New Pet")
        console.print("[d] Delete a Pet")
        console.print("[0] Exit")
        
        choice = console.input("\n[bold cyan]Select choice: [/bold cyan]").strip().lower()
        if choice == "0":
            sys.exit(0)
        elif choice == "n":
            return create_new_pet_flow()
        elif choice == "d":
            delete_pet_flow(files)
        else:
            try:
                num = int(choice)
                if 1 <= num <= len(pet_states):
                    return pet_states[num - 1]
                else:
                    console.print("[red]Invalid selection.[/red]")
            except ValueError:
                console.print("[red]Invalid choice. Enter number, 'n', 'd', or '0'.[/red]")
            time.sleep(1.0)

def create_new_pet_flow():
    console.print("\n=== 🐾 Create a New WeDo 2.0 Desk Pet ===\n", style="bold cyan")
    
    # 1. Profile select
    profiles = ["Puppy", "Kitten", "Robot", "Penguin"]
    table = Table(title="Select a Profile", show_header=False)
    for idx, p in enumerate(profiles):
        table.add_row(f"[{idx + 1}]", p)
    console.print(table)
    
    profile_choice = ""
    while profile_choice not in ["1", "2", "3", "4"]:
        profile_choice = console.input("[cyan]Choose profile (1-4): [/cyan]").strip()
    profile = profiles[int(profile_choice) - 1]
    
    # Default names
    default_names = {
        "Puppy": "Kepler",
        "Kitten": "Luna",
        "Robot": "RoboPet",
        "Penguin": "Pingu"
    }

    
    # 2. Name select
    name = ""
    default_name = default_names[profile]
    while not name:
        name = console.input(f"[cyan]Enter pet name (default '{default_name}'): [/cyan]").strip()
        if not name:
            name = default_name
            
    # Check if duplicate name
    dir_path = os.path.expanduser("~/.wedo_pets")
    if os.path.exists(os.path.join(dir_path, f"{name}.json")):
        console.print(f"[yellow]A pet named '{name}' already exists. Appending a random number...[/yellow]")
        import random
        name = f"{name}{random.randint(10, 99)}"
        
    state = {
        "pet_name": name,
        "profile": profile,
        "level": 1,
        "xp": 0,
        "energy": 80,
        "happiness": 70,
        "hunger": 30,
        "trainer_hp": 100
    }
    
    # Save the initial pet json
    try:
        path = os.path.join(dir_path, f"{name}.json")
        with open(path, "w") as f:
            json.dump(state, f)
        console.print(f"[green]Successfully created '{name}' ({profile})![/green]\n")
        time.sleep(1.0)
    except Exception as e:
        console.print(f"[red]Error saving new pet: {e}[/red]")
        
    return state

def delete_pet_flow(files):
    console.print("\n--- Delete a Saved Pet ---", style="bold red")
    for idx, f in enumerate(sorted(files)):
        console.print(f"[{idx + 1}] {f[:-5]}")
    console.print("[0] Cancel")
    
    choice = console.input("\n[red]Choose pet to delete: [/red]").strip()
    if choice == "0" or not choice:
        return
        
    try:
        num = int(choice)
        if 1 <= num <= len(files):
            filename = sorted(files)[num - 1]
            pet_name = filename[:-5]
            confirm = console.input(f"[red]Are you sure you want to delete '{pet_name}'? (y/n): [/red]").strip().lower()
            if confirm == "y":
                os.remove(os.path.expanduser(f"~/.wedo_pets/{filename}"))
                soul_path = os.path.expanduser(f"~/.wedo_pets/{pet_name}_soul.txt")
                if os.path.exists(soul_path):
                    os.remove(soul_path)
                console.print(f"[green]Deleted '{pet_name}' successfully.[/green]")
        else:
            console.print("[red]Invalid selection.[/red]")
    except Exception as e:
        console.print(f"[red]Delete error: {e}[/red]")
    time.sleep(1.0)


# -----------------------------------------------------------------
# Bluetooth Hub Selection CLI Menu
# -----------------------------------------------------------------
def select_hub_flow(default_hub_name):
    if not ble_available:
        console.print("[yellow]Warning: bleak library is not installed. BLE scanning unavailable.[/yellow]")
        console.print("[yellow]Automatically falling back to Simulated Mock Hub.[/yellow]")
        time.sleep(1.5)
        return MockWeDo2Hub("Mock Smart Hub"), "Simulated (Mock)"

    while True:
        console.print("\n[cyan]Scanning for nearby WeDo Smarthub Bluetooth Low Energy devices (3 seconds)...[/cyan]")
        devices = []
        try:
            devices = runner.run(BleakScanner.discover(timeout=3.0))
        except Exception as e:
            console.print(f"[red]BLE Scan Error: {e}[/red]")
            
        # Filter WeDo-like devices first
        wedo_devices = []
        other_devices = []
        for dev in devices:
            name = dev.name or "Unknown Device"
            if "hub" in name.lower() or "lpf" in name.lower() or "wedo" in name.lower() or "lego" in name.lower() or "smart" in name.lower():
                wedo_devices.append(dev)
            elif dev.name:
                other_devices.append(dev)
                
        display_devices = wedo_devices + other_devices
        
        table = Table(title="📡 Select Your LEGO Smarthub 📡", border_style="cyan")
        table.add_column("No.", style="yellow", justify="right")
        table.add_column("Name", style="green")
        table.add_column("MAC Address", style="magenta")
        table.add_column("RSSI", style="blue")
        
        for idx, dev in enumerate(display_devices):
            rssi = getattr(dev, "rssi", "N/A")
            if rssi == "N/A" and hasattr(dev, "metadata") and dev.metadata:
                rssi = dev.metadata.get("rssi", "N/A")
            rssi_str = f"{rssi} dBm" if rssi != "N/A" else "N/A"
            table.add_row(str(idx + 1), str(dev.name or "Unknown"), str(dev.address), rssi_str)
            
        console.print(table)
        console.print("[m] Start in Mock/Simulation Mode (Offline)")
        console.print("[r] Rescan for Bluetooth Devices")
        console.print("[0] Exit")
        
        choice = console.input("\n[bold cyan]Select choice: [/bold cyan]").strip().lower()
        if choice == "0":
            sys.exit(0)
        elif choice == "r":
            continue
        elif choice == "m":
            return MockWeDo2Hub("Mock Smart Hub"), "Simulated (Mock)"
        else:
            try:
                num = int(choice)
                if 1 <= num <= len(display_devices):
                    target_dev = display_devices[num - 1]
                    target_name = target_dev.name or target_dev.address
                    console.print(f"[cyan]Connecting to Smarthub '{target_name}'...[/cyan]")
                    try:
                        hub = RealWeDo2Hub(target_name)
                        console.print("[green]Connected successfully![/green]")
                        hub.beep(600, 150)
                        time.sleep(0.1)
                        hub.beep(850, 200)
                        hub.set_led("green")
                        return hub, "Physical (BLE)"
                    except Exception as e:
                        console.print(f"[red]Connection failed: {e}[/red]")
                        console.print("Please verify the hub is powered on and retry.")
                        time.sleep(2.0)
                else:
                    console.print("[red]Invalid selection number.[/red]")
                    time.sleep(1.0)
            except ValueError:
                console.print("[red]Invalid choice. Enter a number, 'm', 'r', or '0'.[/red]")
                time.sleep(1.0)


# -----------------------------------------------------------------
# WeDo Games & Interactive Tutorial
# -----------------------------------------------------------------
def handle_games_menu(pet):
    while True:
        console.clear()
        table = Table(title="🎮 WeDo 2.0 Pet Arcade & Games Menu 🎮", border_style="bold green")
        table.add_column("No.", style="yellow", justify="right")
        table.add_column("Game Mode", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("1", "Interactive Live Tutorial", "Step-by-step walk through sensors and triggers")
        table.add_row("2", "Obstacle Course (Stop-Before-Crash)", "Autonomous navigation game with distance sensor")
        table.add_row("3", "Color Simon Says", "Memorize LED color patterns and repeat them")
        table.add_row("0", "Back to Main Menu", "Return to home menu")
        console.print(table)
        
        choice = console.input("\n[bold cyan]Choose game option: [/bold cyan]").strip()
        if choice == "0":
            break
        elif choice == "1":
            run_interactive_tutorial(pet)
        elif choice == "2":
            run_obstacle_course_game(pet)
        elif choice == "3":
            run_simon_says_game(pet)
        else:
            console.print("[red]Invalid selection.[/red]")
            time.sleep(1.0)

def run_interactive_tutorial(pet):
    console.clear()
    console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
    console.print("This live tutorial will guide you through WeDo 2.0 sensors and behaviors.\n")
    
    # Step 1: Petting (Distance Sensor)
    console.print("[yellow]Step 1: Direct Petting Challenge[/yellow]")
    console.print("Put your hand or an object within [bold cyan]6cm[/bold cyan] of the Distance Sensor to pet Kepler.")
    console.print("Waiting for sensor input...")
    
    start_t = time.time()
    pet_detected = False
    while time.time() - start_t < 15.0:
        dist = 10
        dist_p = pet.hub.check_connected("Distance Sensor")
        if dist_p:
            dist = pet.hub.sensor_cache[dist_p]["distance"]
        if dist < 6:
            pet_detected = True
            break
        time.sleep(0.1)
        
    if pet_detected:
        console.print("\n[green]Success! Petting detected! Kepler wagged his tail and chirped happily.[/green]\n")
        pet.interact_pet()
    else:
        console.print("\n[red]Timeout: No petting detected. (Make sure distance sensor is plugged into Port 1 or 2).[/red]\n")
    console.input("Press Enter to proceed to Step 2...")
    
    # Step 2: Dizziness (Tilt Sensor)
    console.clear()
    console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
    console.print("[yellow]Step 2: Tilt Dizziness Demonstration[/yellow]")
    console.print("Tilt the Smarthub in [bold cyan]any direction[/bold cyan] (Forward, Backward, Left, or Right).")
    console.print("Waiting for tilt input...")
    
    start_t = time.time()
    tilt_detected = False
    while time.time() - start_t < 15.0:
        tilt = "Neutral"
        tilt_p = pet.hub.check_connected("Tilt Sensor")
        if tilt_p:
            tilt = pet.hub.sensor_cache[tilt_p]["tilt"]
        if tilt != "Neutral" and tilt != "Unknown":
            tilt_detected = True
            break
        time.sleep(0.1)
        
    if tilt_detected:
        console.print(f"\n[green]Success! Tilt '{tilt}' detected. Kepler is now dizzy! (@_@)[/green]\n")
        for _ in range(4):
            try:
                pet.hub.beep(400, 100)
                time.sleep(0.05)
                pet.hub.beep(300, 100)
                time.sleep(0.05)
            except Exception:
                pass
    else:
        console.print("\n[red]Timeout: No tilt detected. (Make sure tilt sensor is plugged into Port 1 or 2).[/red]\n")
    console.input("Press Enter to proceed to Step 3...")
    
    # Step 3: Feeding Challenge
    console.clear()
    console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
    console.print("[yellow]Step 3: Feeding Challenge Demonstration[/yellow]")
    console.print("Let's simulate a hunger fit! The LED light will turn RED. You must wait for it to turn GREEN,")
    console.print("then click the physical button (or simulate click using option 'b').")
    console.input("Press Enter to start simulation...")
    
    pet.screaming_for_food = True
    pet.screaming_cycle_start = time.time()
    
    while pet.screaming_for_food and pet.is_running:
        time.sleep(0.2)
        
    console.print("\n[green]Great job! You finished the interactive tutorial. Kepler is fully trained![/green]")
    console.input("\nPress Enter to return to games menu...")

def run_obstacle_course_game(pet):
    console.clear()
    console.print("=== 🚗 Obstacle Course Avoidance Game ===\n", style="bold cyan")
    console.print("Instructions: Position your WeDo robot car on a clear path on your desk.")
    console.print("We will run the motor forward at speed 45.")
    console.print("As soon as the distance sensor detects an obstacle (< 5cm), the robot will brake, beep,")
    console.print("and reverse at speed -50 for 1.2 seconds to avoid crashing!")
    console.print("Press Ctrl+C or type 'exit' during the game to abort.")
    console.input("\nPress Enter to start driving...")
    
    score = 0
    driving = True
    try:
        while driving:
            console.print(f"[green]Driving forward... Score (Avoided obstacles): {score}[/green]")
            try:
                pet.hub.set_motor(45)
            except Exception:
                pass
                
            obstacle_avoided = False
            start_check = time.time()
            while time.time() - start_check < 6.0:
                dist = 10
                dist_p = pet.hub.check_connected("Distance Sensor")
                if dist_p:
                    dist = pet.hub.sensor_cache[dist_p]["distance"]
                if dist <= 5:
                    obstacle_avoided = True
                    break
                time.sleep(0.1)
                
            if obstacle_avoided:
                console.print(f"[yellow]⚠️ Obstacle detected at {dist}cm! Braking and reversing![/yellow]")
                try:
                    pet.hub.stop_motor()
                    pet.hub.beep(1000, 100)
                    time.sleep(0.05)
                    pet.hub.beep(1200, 150)
                    pet.hub.set_motor(-50)
                    time.sleep(1.2)
                    pet.hub.stop_motor()
                except Exception:
                    pass
                score += 1
                pet.gain_xp(15)
                
                cont = console.input("\nAvoided! Continue driving? (y/n): ").strip().lower()
                if cont == "n":
                    driving = False
            else:
                console.print("[yellow]Clear path! No obstacles detected for 6 seconds. Pausing...[/yellow]")
                try:
                    pet.hub.stop_motor()
                except Exception:
                    pass
                cont = console.input("Continue driving? (y/n): ").strip().lower()
                if cont == "n":
                    driving = False
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
        except Exception:
            pass
    console.print(f"\n[green]Game Over! Total obstacles avoided: {score}. Total XP gained: {score * 15}![/green]")
    console.input("\nPress Enter to return to games menu...")

def run_simon_says_game(pet):
    console.clear()
    console.print("=== 🔴 Simon Says Color Game 🟢 ===\n", style="bold cyan")
    console.print("Kepler will show you a sequence of LED colors (Red, Green, Blue).")
    console.print("You must memorize the sequence and type the colors in order (e.g. 'rgb' or 'gbr').")
    console.print("Each correct sequence gains Kepler +20 XP!")
    console.input("\nPress Enter to begin Round 1...")
    
    colors_map = {
        "r": ("red", 500),
        "g": ("green", 700),
        "b": ("blue", 900)
    }
    
    round_num = 1
    playing = True
    while playing:
        import random
        seq = [random.choice(["r", "g", "b"]) for _ in range(round_num + 1)]
        
        console.print(f"\n[yellow]Round {round_num}: Watch Kepler's LED light...[/yellow]")
        time.sleep(1.0)
        
        for char in seq:
            col_name, freq = colors_map[char]
            try:
                pet.hub.set_led(col_name)
                pet.hub.beep(freq, 300)
            except Exception:
                pass
            time.sleep(0.4)
            try:
                pet.hub.set_led("off")
            except Exception:
                pass
            time.sleep(0.2)
            
        ans = console.input("[cyan]Enter sequence (r=red, g=green, b=blue): [/cyan]").strip().lower()
        ans_clean = "".join([c for c in ans if c in ["r", "g", "b"]])
        
        expected = "".join(seq)
        if ans_clean == expected:
            console.print("[green]Correct! Kepler wags his tail happily![/green]")
            try:
                pet.hub.beep(800, 100)
                pet.hub.beep(1000, 100)
                pet.hub.beep(1200, 150)
            except Exception:
                pass
            pet.gain_xp(20)
            round_num += 1
            cont = console.input("Proceed to next round? (y/n): ").strip().lower()
            if cont == "n":
                playing = False
        else:
            console.print(f"[red]Wrong! Expected sequence was: {expected.upper()}[/red]")
            try:
                pet.hub.beep(200, 400)
            except Exception:
                pass
            playing = False
            
    console.print(f"\n[green]Game Over! Reached Round {round_num}. Kepler is proud of you![/green]")
    console.input("\nPress Enter to return to games menu...")


# -----------------------------------------------------------------
# Ollama Setup Installer helpers
# -----------------------------------------------------------------
def handle_ollama_setup(pet):
    while True:
        console.clear()
        console.print("=== 🧠 Ollama Local LLM AI Setup Helper ===\n", style="bold cyan")
        
        running = False
        import urllib.request
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1) as r:
                if r.status == 200:
                    running = True
        except Exception:
            pass
            
        status_str = "[bold green]ONLINE (Running)[/bold green]" if running else "[bold red]OFFLINE (Not Detected)[/bold red]"
        console.print(f"Ollama Server Status: {status_str}\n")
        
        console.print("This helper allows you to automatically install Ollama or pull models directly.")
        console.print("[1] Install Ollama (Linux install script via curl)")
        console.print("[2] Download / Pull qwen2.5:3b Model (Required for brain)")
        console.print("[3] Test Chat with Ollama")
        console.print("[0] Return to Main Menu")
        
        choice = console.input("\n[bold cyan]Choose option: [/bold cyan]").strip()
        if choice == "0":
            break
        elif choice == "1":
            install_ollama_command()
        elif choice == "2":
            pull_qwen_command()
        elif choice == "3":
            if not running:
                console.print("[red]Error: Ollama server is offline. Please start it first.[/red]")
                time.sleep(1.5)
                continue
            console.print("[cyan]Connecting to qwen2.5:3b model...[/cyan]")
            res = pet.query_ollama("Hello, test beep command!")
            if res:
                console.print(f"[green]Success! Responded: {res.get('speech')}[/green]")
            else:
                console.print("[red]Failed: Did you pull 'qwen2.5:3b' yet?[/red]")
            console.input("\nPress Enter to continue...")

def install_ollama_command():
    console.print("\n[cyan]Starting Ollama installation... (requires sudo and curl)[/cyan]")
    console.print("Running command: curl -fsSL https://ollama.com/install.sh | sh")
    console.print("[yellow]Please review and approve command execution when prompted by system...[/yellow]\n")
    import subprocess
    try:
        proc = subprocess.Popen("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        proc.wait()
        console.print("\n[green]Ollama installation process completed.[/green]")
    except Exception as e:
        console.print(f"[red]Error starting installer: {e}[/red]")
    console.input("\nPress Enter to return...")

def pull_qwen_command():
    console.print("\n[cyan]Starting: ollama pull qwen2.5:3b[/cyan]")
    console.print("This will download the 1.9GB model. Please wait, this might take a few minutes...")
    import subprocess
    try:
        proc = subprocess.Popen(["ollama", "pull", "qwen2.5:3b"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
        if proc.returncode == 0:
            console.print("\n[green]Successfully downloaded qwen2.5:3b model![/green]")
        else:
            console.print(f"\n[red]Failed to pull model. Return code: {proc.returncode}[/red]")
    except Exception as e:
        console.print(f"[red]Error starting ollama pull: {e}[/red]")
    console.input("\nPress Enter to return...")


# -----------------------------------------------------------------
# CLI Connection Launcher & Arguments
# -----------------------------------------------------------------
def main():



    parser = argparse.ArgumentParser(description="Start the interactive LEGO WeDo 2.0 CLI Desk Pet.")
    parser.add_argument("--hub-name", type=str, default="Isaiah Smart Hub",
                        help="Bluetooth name of your WeDo 2.0 hub (default: Isaiah Smart Hub).")

    parser.add_argument("--mock", action="store_true",
                        help="Run in simulation/mock mode without connecting to physical hardware.")
    parser.add_argument("--scan", action="store_true",
                        help="Scan and list nearby BLE devices, then exit.")
    args = parser.parse_args()

    if args.scan:
        if not ble_available:
            console.print("[red]Error: bleak is not installed. BLE scan unavailable.[/red]")
            sys.exit(1)
        console.print("[cyan]Scanning for nearby Bluetooth Low Energy devices (5 seconds)...[/cyan]")
        devices = runner.run(BleakScanner.discover(timeout=5.0))
        table = Table(title="Nearby Bluetooth Devices")
        table.add_column("Name", style="cyan")
        table.add_column("MAC Address", style="magenta")
        table.add_column("RSSI", style="green")
        for dev in devices:
            rssi = getattr(dev, "rssi", "N/A")
            if rssi == "N/A" and hasattr(dev, "metadata") and dev.metadata:
                rssi = dev.metadata.get("rssi", "N/A")
            table.add_row(str(dev.name or "Unknown"), str(dev.address), f"{rssi} dBm" if rssi != "N/A" else "N/A")
        console.print(table)
        sys.exit(0)


    # Resolve Hub Connection
    hub = None
    hub_type = "Physical (BLE)"
    
    if args.mock:
        console.print("[yellow]Starting in MOCK / SIMULATION mode as requested.[/yellow]")
        hub = MockWeDo2Hub(args.hub_name)
        hub_type = "Simulated (Mock)"
    else:
        hub, hub_type = select_hub_flow(args.hub_name)


    # Select or initialize Pet state
    state_dict = select_or_create_pet()
    pet = DeskPet(hub, state_dict)


    # Show initial live dashboard
    run_live_dashboard(pet, hub_type)

    # Main Interactive CLI Command Loop
    try:
        while pet.is_running:
            print_main_menu(pet)
            choice = console.input("[bold green]Choose an action (0-9, a, c, b, g, o, t): [/bold green]").strip().lower()


            if choice == "1":
                run_live_dashboard(pet, hub_type)
            elif choice == "2":
                pet.interact_feed()
            elif choice == "3":
                pet.interact_pet()
            elif choice == "4":
                pet.interact_poke()
            elif choice == "5":
                console.print("[cyan]Playing chime melody...[/cyan]")
                scale = [523, 587, 659, 698, 784, 880, 988, 1047]
                for freq in scale:
                    hub.beep(freq, 120)
                    time.sleep(0.08)
                pet.add_log("Played chime melody.")
            elif choice == "6":
                handle_music_center(pet)
            elif choice == "7":
                handle_tuning_menu(pet)
            elif choice == "8":
                handle_profile_menu(pet)
            elif choice == "9":
                dist_p = hub.check_connected("Distance Sensor")
                tilt_p = hub.check_connected("Tilt Sensor")
                dist_str = f"{hub.sensor_cache[dist_p]['distance']} cm" if dist_p else "Not Connected"
                tilt_str = f"{hub.sensor_cache[tilt_p]['tilt']}" if tilt_p else "Not Connected"
                
                console.print("\n=== WeDo Smarthub Diagnostics ===", style="bold cyan")
                console.print(f"Connection Status: {'Connected' if hub.connected_state else 'Disconnected'}")
                console.print(f"Hub BLE Name: {hub.hub_name}")
                console.print(f"Battery: {hub.get_battery_level()}%")
                console.print(f"Connected Ports Data: {hub.port_data}")
                console.print(f"Distance Sensor: {dist_str}")
                console.print(f"Tilt Sensor: {tilt_str}")
                console.print(f"Pet State: Mood: {pet.mood}, Energy: {pet.energy}, Hunger: {pet.hunger}, Happiness: {pet.happiness}")
                console.input("\nPress Enter to continue...")
            elif choice == "a":
                pet.ai_autopilot = not pet.ai_autopilot
                pet.add_log(f"AI Autopilot mode set to {pet.ai_autopilot}")
                console.print(f"[green]AI Autopilot mode set to: {'ON' if pet.ai_autopilot else 'OFF'}[/green]")
                time.sleep(1.0)
            elif choice == "c":
                handle_chat_mode(pet)
            elif choice == "g":
                handle_games_menu(pet)
            elif choice == "o":
                handle_ollama_setup(pet)
            elif choice == "t":
                handle_training_manual()
            elif choice == "b" and "(MOCK)" in hub.hub_name:
                def simulate_click():
                    hub.button_state = 1
                    time.sleep(0.5)
                    hub.button_state = 0
                threading.Thread(target=simulate_click, daemon=True).start()
                console.print("[yellow]Simulating physical button click on WeDo Smarthub...[/yellow]")
                time.sleep(0.6)


            elif choice == "0":
                console.print("[cyan]Disconnecting from LEGO Smarthub...[/cyan]")
                break
            else:
                console.print("[red]Invalid selection. Please choose a valid action.[/red]")
                time.sleep(0.8)


                
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Exiting cleanly...[/yellow]")
    finally:
        pet.is_running = False
        hub.stop_motor()
        try:
            hub.set_led("blue")
            time.sleep(0.2)
        except Exception:
            pass
        hub.disconnect()
        console.print("[green]Goodbye![/green]")

if __name__ == "__main__":
    main()
