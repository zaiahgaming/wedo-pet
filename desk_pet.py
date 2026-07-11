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

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None

GAMES_LIST = {
    "tutorial": ("Interactive Live Tutorial", None),
    "hide_seek": ("🙈 Hide & Seek", "run_hide_and_seek_game"),
    "speed_pet": ("⏱️ Speed Petting Sprint", "run_speed_petting_game"),
    "rhythm": ("🥁 Tail-Wag Rhythm Matcher", "run_rhythm_matcher_game"),
    "balance": ("⚖️ Balance the Tail", "run_balance_tail_game"),
    "sound_mem": ("🎵 Sound Pitch Memory", "run_sound_memory_game"),
    "code_break": ("🔐 Pet Code Breaker", "run_code_breaker_game"),
    "snatcher": ("⚡ Tail Snatcher", "run_tail_snatcher_game"),
    "dj": ("🎧 Sound DJ", "run_sound_dj_game"),
    "maze": ("🌀 Tilt Maze Navigator", "run_tilt_maze_game"),
    "keep_away": ("🛡️ Keep-Away", "run_keep_away_game"),
    "simon_tilt": ("📐 Simon Says: Tilt", "run_simon_says_tilt_game"),
    "simon_color": ("🔴 Color Simon Says", "run_simon_says_game"),
    "tail_counter": ("🧮 Tail Counter", "run_tail_counter_game"),
    "tug_of_war": ("🪢 Pet Tug-of-War", "run_tug_of_war_game")
}

import math

class SmallBrainNN:
    def __init__(self):
        # 12 inputs -> 8 hidden -> 7 outputs
        self.input_size = 12
        self.hidden_size = 8
        self.output_size = 7
        
        # Xavier/Glorot initialization
        self.W1 = [[random.uniform(-1.0, 1.0) * math.sqrt(2.0/self.input_size) 
                    for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.b1 = [0.0] * self.hidden_size
        
        self.W2 = [[random.uniform(-1.0, 1.0) * math.sqrt(2.0/self.hidden_size) 
                    for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        self.b2 = [0.0] * self.output_size

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def softmax(self, arr):
        max_val = max(arr)
        exps = [math.exp(x - max_val) for x in arr]
        sum_exps = sum(exps)
        return [x / sum_exps for x in exps]

    def forward(self, X):
        self.h = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            val = sum(X[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j]
            self.h[j] = self.sigmoid(val)
            
        out = [0.0] * self.output_size
        for j in range(self.output_size):
            val = sum(self.h[i] * self.W2[i][j] for i in range(self.hidden_size)) + self.b2[j]
            out[j] = val
            
        return self.softmax(out)

    def train(self, dataset, epochs=500, lr=0.1):
        for _ in range(epochs):
            for X, target in dataset:
                outputs = self.forward(X)
                d_out = [outputs[i] - target[i] for i in range(self.output_size)]
                
                d_hidden = [0.0] * self.hidden_size
                for i in range(self.hidden_size):
                    error = sum(d_out[j] * self.W2[i][j] for j in range(self.output_size))
                    d_hidden[i] = error * self.h[i] * (1.0 - self.h[i])
                    
                for i in range(self.hidden_size):
                    for j in range(self.output_size):
                        self.W2[i][j] -= lr * self.h[i] * d_out[j]
                for j in range(self.output_size):
                    self.b2[j] -= lr * d_out[j]
                    
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.W1[i][j] -= lr * X[i] * d_hidden[j]
                for j in range(self.hidden_size):
                    self.b1[j] -= lr * d_hidden[j]

def train_default_brain():
    brain = SmallBrainNN()
    dataset = [
        ([1, 0, 0, 0, 0, 0,  1.0, 0,  0.5, 0.5, 0.8, 1.0], [0, 0, 0, 0, 0, 1, 0]),
        ([1, 0, 0, 0, 0, 0,  0.5, 0,  0.2, 0.3, 0.9, 1.0], [0, 0, 0, 0, 0, 1, 0]),
        ([0, 1, 0, 0, 0, 0,  0.1, 0,  0.8, 0.8, 0.2, 1.0], [1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0,  1.0, 0,  0.9, 0.9, 0.2, 1.0], [1, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0,  1.0, 0,  0.5, 0.2, 0.3, 1.0], [0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0,  0.8, 0,  0.6, 0.3, 0.4, 0.8], [0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0,  1.0, 0,  0.8, 0.8, 0.2, 1.0], [0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0,  1.0, 0,  0.6, 0.7, 0.3, 1.0], [0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0,  1.0, 0,  0.1, 0.2, 0.2, 1.0], [0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0,  1.0, 0,  0.05, 0.1, 0.3, 1.0], [0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0,  1.0, 1,  0.7, 0.7, 0.3, 1.0], [0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 0, 0, 1,  1.0, 1,  0.8, 0.5, 0.4, 1.0], [0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 0, 0, 0,  1.0, 0,  0.8, 0.8, 0.2, 1.0], [0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0,  1.0, 0,  0.5, 0.5, 0.5, 1.0], [0, 0, 0, 0, 0, 0, 1]),
    ]
    brain.train(dataset, epochs=500, lr=0.15)
    return brain


_terminal_is_raw = False
_old_terminal_settings = None

def set_terminal_raw(raw=True):
    global _terminal_is_raw, _old_terminal_settings
    if sys.platform.startswith("win"):
        return
    import tty
    import termios
    fd = sys.stdin.fileno()
    if raw:
        if not _terminal_is_raw:
            _old_terminal_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            _terminal_is_raw = True
    else:
        if _terminal_is_raw and _old_terminal_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, _old_terminal_settings)
            _terminal_is_raw = False

def read_key(timeout=10.0):
    global _terminal_is_raw, _old_terminal_settings
    if sys.platform.startswith("win"):
        import msvcrt
        start_t = time.time()
        while time.time() - start_t < timeout:
            if msvcrt.kbhit():
                try:
                    ch = msvcrt.getch()
                    if ch in (b'\x00', b'\xe0'):
                        ch2 = msvcrt.getch()
                        if ch2 == b'H': return "up"
                        if ch2 == b'P': return "down"
                        if ch2 == b'K': return "left"
                        if ch2 == b'M': return "right"
                    if ch == b'\r': return "enter"
                    if ch == b'\x1b': return "escape"
                    if ch == b'\x03': raise KeyboardInterrupt()
                    return ch.decode("utf-8").lower()
                except Exception:
                    return ""
            time.sleep(0.01)
        return ""
    else:
        import select
        fd = sys.stdin.fileno()
        was_raw = _terminal_is_raw
        if not was_raw:
            set_terminal_raw(True)
        try:
            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return ""
            ch_bytes = os.read(fd, 1)
            if not ch_bytes:
                return ""
            ch = ch_bytes[0]
            if ch == 3:
                raise KeyboardInterrupt()
            if ch == 27:
                r_esc, _, _ = select.select([fd], [], [], 0.05)
                if r_esc:
                    ch2_bytes = os.read(fd, 1)
                    if ch2_bytes and ch2_bytes[0] == 91:
                        r_seq, _, _ = select.select([fd], [], [], 0.05)
                        if r_seq:
                            ch3_bytes = os.read(fd, 1)
                            if ch3_bytes:
                                ch3 = ch3_bytes[0]
                                if ch3 == 65: return "up"
                                if ch3 == 66: return "down"
                                if ch3 == 67: return "right"
                                if ch3 == 68: return "left"
                return "escape"
            if ch in (13, 10):
                return "enter"
            try:
                return chr(ch).lower()
            except Exception:
                return ""
        finally:
            if not was_raw:
                set_terminal_raw(False)

def choose_option_interactive(title, options, prompt_message="Use Up/Down arrows and press Enter to select:"):
    idx = 0
    while True:
        console.clear()
        console.print(Panel(Text(f"🐾 {title} 🐾", style="bold green", justify="center"), border_style="green"))
        
        for i, opt in enumerate(options):
            if i == idx:
                console.print(Text(f"  ▶  {opt}", style="bold #00E676"))
            else:
                console.print(Text(f"     {opt}", style="dim"))
                
        console.print(f"\n[dim]{prompt_message}[/dim]")
        
        key = read_key()
        if key == "up":
            idx = (idx - 1) % len(options)
        elif key == "down":
            idx = (idx + 1) % len(options)
        elif key == "enter":
            return idx
        elif key == "escape":
            return -1
def print_main_menu_interactive(pet, selected_idx):
    from rich.align import Align
    
    face_art = pet.get_face()
    
    e_color = "#00E676" if pet.energy > 50 else "#FFEB3B" if pet.energy > 20 else "#FF1744"
    h_color = "#00E676" if pet.happiness > 50 else "#FFEB3B" if pet.happiness > 20 else "#FF1744"
    hu_color = "#FF1744" if pet.hunger > 60 else "#FFEB3B" if pet.hunger > 30 else "#00E676"
    
    dist_p = pet.hub.check_connected("Distance Sensor")
    tilt_p = pet.hub.check_connected("Tilt Sensor")
    
    if dist_p:
        dist_val = pet.hub.sensor_cache[dist_p]["distance"]
        dist_str = f"[bold #FF9100]{'█' * int(dist_val)}[/bold #FF9100] [dim]{dist_val} cm[/dim]"
    else:
        dist_str = "[#888888]Not Connected[/#888888]"
        
    tilt_val = None
    if tilt_p:
        tilt_val = pet.hub.sensor_cache[tilt_p]["tilt"]
        
    tilt_icons = {
        "Neutral": "● Neutral",
        "Left": "◀ Left",
        "Right": "▶ Right",
        "Forward": "▲ Forward",
        "Backward": "▼ Backward",
        "Unknown": "? Unknown"
    }
    tilt_str = f"[bold #FF9100]{tilt_icons.get(tilt_val, tilt_val)}[/bold #FF9100]" if tilt_val else "[#888888]Not Connected[/#888888]"
    
    xp_needed = pet.get_xp_needed()
    xp_pct = min(100, max(0, int((pet.xp / xp_needed) * 100)))
    ai_status = "[bold #00E676]ON[/bold #00E676]" if pet.ai_autopilot else "[bold #FF1744]OFF[/bold #FF1744]"
    
    def make_bar(val, color_hex):
        filled = min(10, max(0, int(val / 10)))
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        return f"[bold {color_hex}]{bar}[/bold {color_hex}] [bold]{val}/100[/bold]"

    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="#00BBFF", justify="right")
    status_table.add_column(style="white")
    status_table.add_row("Profile: ", f"[bold #9B5DE5]{pet.profile}[/bold #9B5DE5] ({pet.pet_name})")
    status_table.add_row("Level: ", f"[bold #FFD600]{pet.level}[/bold #FFD600] ({pet.get_level_title()})")
    status_table.add_row("XP: ", make_bar(xp_pct, "#9B5DE5") + f" [dim]{pet.xp}/{xp_needed} XP[/dim]")
    status_table.add_row("State/Mood: ", f"[bold #F15BB5]{pet.mood.upper()}[/bold #F15BB5]")
    status_table.add_row("Energy: ", make_bar(pet.energy, e_color))
    status_table.add_row("Happiness: ", make_bar(pet.happiness, h_color))
    status_table.add_row("Hunger: ", make_bar(pet.hunger, hu_color))
    status_table.add_row("Distance: ", dist_str)
    status_table.add_row("Tilt: ", tilt_str)
    status_table.add_row("AI Autopilot: ", ai_status)
    status_table.add_row("Trainer HP: ", make_bar(pet.trainer_hp, "#FF1744" if pet.trainer_hp < 40 else "#00E676"))

    face_panel = Panel(
        Align.center(Text(f"\n\n  {face_art}  \n\n", style="bold #FFD600", justify="center")),
        title="Pet Expression",
        border_style="#FFD600"
    )
    
    left_layout = Layout()
    left_layout.split_column(
        Layout(face_panel, ratio=4),
        Layout(Panel(status_table, title="Vital Telemetry", border_style="#00E676"), ratio=6)
    )
    
    menu_options = [
        "Enter Live Dashboard Mode",
        "Feed Pet",
        "Pet the Pet",
        "Poke Pet",
        "Sing a Custom Melody",
        "Music Center: Play MIDI Song",
        "Tuning & Manual Hardware Overrides",
        f"Change Pet Profile [Current: {pet.profile}]",
        "Hub Status & Telemetry Summary",
        "Play Games & Live Tutorial",
        "Ollama Local LLM Setup Helper",
        f"Toggle Autopilot Mode [Currently: {'ON' if pet.ai_autopilot else 'OFF'}]",
        "Chat with Pet (Ollama AI)",
        "User Training & Manual",
    ]
    if "(MOCK)" in pet.hub.hub_name:
        menu_options.append("Simulate Hub Button Click")
    menu_options.append("Exit (Disconnect & Close)")

    menu_text = Text()
    for i, opt in enumerate(menu_options):
        if i == selected_idx:
            menu_text.append(f" ▶  {opt}\n", style="bold #00E676")
        else:
            menu_text.append(f"     {opt}\n", style="dim")
            
    right_panel = Panel(menu_text, title="Action Menu", border_style="green")
    
    main_layout = Layout()
    main_layout.split_row(
        Layout(left_layout, ratio=4),
        Layout(right_panel, ratio=5)
    )
    
    console.clear()
    console.print(Panel(Text("🐾 WeDo 2.0 Desk Pet Dashboard & Controller 🐾", style="bold cyan", justify="center"), border_style="cyan"))
    console.print(main_layout)
    console.print("\n[dim]Use Up/Down Arrow keys to navigate options, and press Enter to select.[/dim]")



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
                        raw_dist = DISTANCE_LOOKUP.get(val_hex, 0)
                        history = self.sensor_cache[port].setdefault("history", [])
                        history.append(raw_dist)
                        if len(history) > 3:
                            history.pop(0)
                        sorted_hist = sorted(history)
                        median_val = sorted_hist[len(sorted_hist) // 2]
                        self.sensor_cache[port]["distance"] = median_val
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
        if getattr(self, "music_playing", False):
            return
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
        if getattr(self, "music_playing", False):
            return
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

        # Local Neural Network brain
        self.local_brain_nn = train_default_brain()
        self.wants_game_id = None

        self.add_log(f"Pet {self.pet_name} initialized. Welcome!")

        # Setup SIGHUP/SIGBREAK signal handlers to detach terminal and run in background
        import signal
        def handle_detach(signum, frame):
            self.add_log("Terminal window closed! Detaching and running in background system tray...")
            try:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
            except Exception:
                pass

        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, handle_detach)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, handle_detach)

        # Start the background system tray icon thread if pystray is available
        if pystray:
            threading.Thread(target=self._start_system_tray, daemon=True).start()

        # Start autonomous behavior thread
        self.brain_thread = threading.Thread(target=self._brain_loop, daemon=True)
        self.brain_thread.start()

    def get_available_games(self):
        dist_p = self.hub.check_connected("Distance Sensor")
        tilt_p = self.hub.check_connected("Tilt Sensor")
        motor_p = self.hub.check_connected("Motor")
        
        avail = ["tutorial", "sound_mem", "simon_color"]
        if dist_p and motor_p:
            avail.extend(["hide_seek", "speed_pet", "snatcher", "keep_away", "tail_counter"])
        if motor_p:
            avail.append("rhythm")
        if tilt_p and motor_p:
            avail.extend(["balance", "maze", "tug_of_war"])
        if dist_p and tilt_p:
            avail.append("code_break")
        if dist_p:
            avail.append("dj")
        if tilt_p:
            avail.append("simon_tilt")
        return avail

    def _start_system_tray(self):
        def create_tray_image():
            image = Image.new('RGB', (64, 64), color=(30, 30, 30))
            draw = ImageDraw.Draw(image)
            draw.ellipse((8, 8, 56, 56), fill=(0, 230, 118))
            draw.ellipse((22, 22, 42, 42), fill=(255, 255, 255))
            return image

        def on_show_dashboard(icon, item):
            import platform
            import subprocess
            script_path = os.path.abspath(sys.argv[0])
            try:
                if platform.system() == "Linux":
                    try:
                        subprocess.Popen(["gnome-terminal", "--", script_path, "--pet-name", self.pet_name])
                    except Exception:
                        subprocess.Popen(["xterm", "-e", script_path, "--pet-name", self.pet_name])
                elif platform.system() == "Windows":
                    subprocess.Popen([script_path, "--pet-name", self.pet_name], shell=True)
                elif platform.system() == "Darwin":
                    cmd = f'tell application "Terminal" to do script "{script_path} --pet-name {self.pet_name}"'
                    subprocess.Popen(["osascript", "-e", cmd])
            except Exception as e:
                self.add_log(f"Error launching terminal: {e}")

        def on_feed(icon, item):
            self.interact_feed()

        def on_pet(icon, item):
            self.interact_pet()

        def on_poke(icon, item):
            self.interact_poke()

        def on_chime(icon, item):
            def play_chime():
                try:
                    scale = [523, 659, 784, 1047]
                    for freq in scale:
                        self.hub.beep(freq, 120)
                        time.sleep(0.08)
                except Exception:
                    pass
            threading.Thread(target=play_chime, daemon=True).start()

        def on_toggle_autopilot(icon, item):
            self.ai_autopilot = not self.ai_autopilot
            self.add_log(f"AI Autopilot mode set to {self.ai_autopilot}")

        def on_profile(profile_name):
            def f(icon, item):
                self.change_profile(profile_name)
            return f

        def run_game_window(game_id):
            def f(icon, item):
                import platform
                import subprocess
                script_path = os.path.abspath(sys.argv[0])
                try:
                    if platform.system() == "Linux":
                        try:
                            subprocess.Popen(["gnome-terminal", "--", script_path, "--play-game", game_id, "--pet-name", self.pet_name])
                        except Exception:
                            subprocess.Popen(["xterm", "-e", script_path, "--play-game", game_id, "--pet-name", self.pet_name])
                    elif platform.system() == "Windows":
                        subprocess.Popen([script_path, "--play-game", game_id, "--pet-name", self.pet_name], shell=True)
                    elif platform.system() == "Darwin":
                        cmd = f'tell application "Terminal" to do script "{script_path} --play-game {game_id} --pet-name {self.pet_name}"'
                        subprocess.Popen(["osascript", "-e", cmd])
                except Exception as e:
                    self.add_log(f"Error launching game terminal: {e}")
            return f

        def get_games_submenu():
            items = []
            avail = self.get_available_games()
            for game_id in avail:
                title = GAMES_LIST[game_id][0]
                items.append(pystray.MenuItem(title, run_game_window(game_id)))
            return pystray.Menu(*items)

        def on_play_headless(icon, item):
            avail = self.get_available_games()
            headless_games = ["hide_seek", "tail_counter", "tug_of_war", "simon_tilt", "dj"]
            playable_headless = [g for g in avail if g in headless_games]
            if playable_headless:
                import random
                chosen = random.choice(playable_headless)
                self.run_background_game(chosen)
                if self.tray_icon:
                    try:
                        self.tray_icon.notify(
                            f"Starting {GAMES_LIST[chosen][0]}! Interact with the physical hub box using sensors/LEDs/beeps!",
                            "Headless Game Started"
                        )
                    except Exception:
                        pass
            else:
                if self.tray_icon:
                    try:
                        self.tray_icon.notify(
                            "Connect a Distance or Tilt sensor to play headless games!",
                            "No Sensors Detected"
                        )
                    except Exception:
                        pass

        def on_exit(icon, item):
            icon.stop()
            self.is_running = False
            try:
                self.hub.stop_motor()
            except Exception:
                pass
            os._exit(0)

        menu = pystray.Menu(
            pystray.MenuItem("Feed Pet", on_feed),
            pystray.MenuItem("Pet Kepler", on_pet),
            pystray.MenuItem("Poke Kepler", on_poke),
            pystray.MenuItem("Play Happy Chime", on_chime),
            pystray.MenuItem("Toggle Autopilot", on_toggle_autopilot, checked=lambda item: self.ai_autopilot),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Play Game (Interactive Window)", get_games_submenu),
            pystray.MenuItem("Play Headless Game (No Screen)", on_play_headless),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Change Profile", pystray.Menu(
                pystray.MenuItem("Puppy (Kepler)", on_profile("Puppy"), checked=lambda item: self.profile == "Puppy"),
                pystray.MenuItem("Kitten (Luna)", on_profile("Kitten"), checked=lambda item: self.profile == "Kitten"),
                pystray.MenuItem("Robot (RoboPet)", on_profile("Robot"), checked=lambda item: self.profile == "Robot"),
                pystray.MenuItem("Penguin (Pingu)", on_profile("Penguin"), checked=lambda item: self.profile == "Penguin")
            )),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Telemetry Dashboard", on_show_dashboard),
            pystray.MenuItem("Exit Pet Application", on_exit)
        )

        self.tray_icon = pystray.Icon(
            "wedo_pet",
            create_tray_image(),
            menu=menu,
            title=f"WeDo Pet: {self.pet_name} (Running)"
        )
        self.tray_icon.run()

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
            self.add_log("[Local Brain] Redirecting query to local Neural Network companion brain...")
            return self.query_local_brain_nn(prompt)

    def query_local_brain_nn(self, prompt):
        p = str(prompt).lower()
        
        # Build 12 inputs: [feed, pet, poke, sing, sleep, play, dist_val, tilt_val, energy, happiness, hunger, hp]
        feed = 1 if any(w in p for w in ["feed", "food", "cookie", "hungry", "eat", "fish"]) else 0
        pet = 1 if any(w in p for w in ["pet", "scratch", "love", "cuddle", "hug", "happy", "cute"]) else 0
        poke = 1 if any(w in p for w in ["poke", "hit", "hurt", "kick", "angry", "punch", "slap"]) else 0
        sing = 1 if any(w in p for w in ["sing", "song", "music", "melody", "sound", "tune"]) else 0
        sleep = 1 if any(w in p for w in ["sleep", "dream", "bed", "night", "rest", "tired"]) else 0
        play = 1 if any(w in p for w in ["play", "game", "dance", "jump", "walk"]) else 0
        
        # Get sensor inputs
        dist_val = 1.0
        dist_port = self.hub.check_connected("Distance Sensor")
        if dist_port:
            dist_val = self.hub.sensor_cache[dist_port]["distance"] / 10.0
            
        tilt_val = 0
        tilt_port = self.hub.check_connected("Tilt Sensor")
        if tilt_port:
            t = self.hub.sensor_cache[tilt_port]["tilt"]
            if t != "Neutral" and t != "Unknown":
                tilt_val = 1
                
        # Forward pass on small brain
        inputs = [
            feed, pet, poke, sing, sleep, play,
            dist_val, float(tilt_val),
            self.energy / 100.0,
            self.happiness / 100.0,
            self.hunger / 100.0,
            self.trainer_hp / 100.0
        ]
        
        outputs = self.local_brain_nn.forward(inputs)
        best_idx = outputs.index(max(outputs))
        
        profile_responses = {
            "Puppy": {
                0: ("Happy!", "Woof woof! Happy tail wagging!", "happy", "green", [[800, 100], [1000, 150]], 40, 200),
                1: ("Angry!", "Grrr! Snarl peevishly!", "angry", "red", [[150, 200], [100, 250]], 80, 200),
                2: ("Singing", "Awoo! Puppy howl melody! ♪", "singing", "purple", [[400, 200], [500, 200], [600, 300]], 0, 0),
                3: ("Sleeping", "Yawn.. Zzz.. Dreaming of bones.", "sleeping", "blue", [[300, 400]], 0, 0),
                4: ("Dizzy", "Woof.. The floor is spinning!", "dizzy", "yellow", [[200, 150], [150, 150]], 0, 0),
                5: ("Eating", "Munch munch! Delicious cookie!", "eating", "green", [[600, 100], [900, 200]], 30, 200),
                6: ("Waiting", "Woof? (Waiting for you)", "awake", "green", [], 0, 0)
            },
            "Kitten": {
                0: ("Happy!", "Meow! Purr purr...", "happy", "green", [[900, 100], [1100, 150]], 30, 150),
                1: ("Angry!", "Hiss! Kitten claws out!", "angry", "red", [[180, 200], [120, 250]], 90, 250),
                2: ("Singing", "Meow meow, cute notes! ♪", "singing", "purple", [[450, 200], [550, 200], [650, 300]], 0, 0),
                3: ("Sleeping", "Prr.. Zzz.. Nap time.", "sleeping", "blue", [[350, 400]], 0, 0),
                4: ("Dizzy", "Meow.. spinning world!", "dizzy", "yellow", [[220, 150], [170, 150]], 0, 0),
                5: ("Eating", "Chomp chomp! Tasty fish treats!", "eating", "green", [[650, 100], [950, 200]], 25, 150),
                6: ("Waiting", "Meow? (Blinking at you)", "awake", "green", [], 0, 0)
            },
            "Robot": {
                0: ("Happy!", "BIP BOOP! System happy!", "happy", "green", [[1000, 100], [1200, 150]], 50, 300),
                1: ("Angry!", "ERROR! System overload!", "angry", "red", [[200, 200], [150, 250]], 100, 300),
                2: ("Singing", "Beep chime, retro wave! ♪", "singing", "purple", [[500, 200], [600, 200], [700, 300]], 0, 0),
                3: ("Sleeping", "HALT.. Power down.. Zzz.", "sleeping", "blue", [[400, 400]], 0, 0),
                4: ("Dizzy", "GYRO ERROR! Unstable axis!", "dizzy", "yellow", [[250, 150], [200, 150]], 0, 0),
                5: ("Eating", "CRUNCH! Recharging cells!", "eating", "green", [[700, 100], [1000, 200]], 40, 300),
                6: ("Waiting", "SYS IDLE (Waiting inputs)", "awake", "green", [], 0, 0)
            },
            "Penguin": {
                0: ("Happy!", "Squawk! Waddle waddle happy!", "happy", "green", [[850, 100], [1050, 150]], 45, 180),
                1: ("Angry!", "Peck peck! Annoyed squawk!", "angry", "red", [[160, 200], [110, 250]], 85, 180),
                2: ("Singing", "Honk honk, penguin tunes! ♪", "singing", "purple", [[420, 200], [520, 200], [620, 300]], 0, 0),
                3: ("Sleeping", "Zzz.. Dreaming of cold ice.", "sleeping", "blue", [[320, 400]], 0, 0),
                4: ("Dizzy", "Squawk.. The glacier is spinning!", "dizzy", "yellow", [[210, 150], [160, 150]], 0, 0),
                5: ("Eating", "Yum! Chasing fish! Chomp!", "eating", "green", [[620, 100], [920, 200]], 35, 180),
                6: ("Waiting", "Squawk? (Looking around)", "awake", "green", [], 0, 0)
            }
        }
        
        resp_dict = profile_responses.get(self.profile, profile_responses["Puppy"])
        thought, speech, emotion, color, sound, motor_speed, motor_duration = resp_dict[best_idx]
        
        response = {
            "thought": f"LocalNN: {thought}",
            "speech": speech,
            "emotion": emotion,
            "color": color,
            "sound": sound,
            "motor_speed": motor_speed,
            "motor_duration_ms": motor_duration,
            "prank_rickroll": False,
            "write_soul": None,
            "vm_code": None
        }
        return response


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

    def execute_goofy_action(self, event):
        def run():
            if "order 66" in event:
                self.hub.set_led("red")
                for _ in range(4):
                    try:
                        self.hub.set_motor(90)
                        self.hub.beep(200, 100)
                        time.sleep(0.1)
                        self.hub.set_motor(-90)
                        time.sleep(0.1)
                    except Exception:
                        pass
                try:
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "divide by zero" in event:
                try:
                    self.hub.set_led("orange")
                    self.hub.beep(300, 150)
                    self.hub.set_motor(50)
                    time.sleep(0.1)
                    self.hub.set_motor(-50)
                    time.sleep(0.1)
                    self.hub.stop_motor()
                    self.hub.set_led("off")
                    self.hub.beep(150, 200)
                except Exception:
                    pass
            elif "vacuum cleaner" in event:
                try:
                    self.hub.set_led("purple")
                    self.hub.beep(1000, 300)
                    self.hub.set_motor(100)
                    time.sleep(0.6)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "poetry" in event:
                try:
                    self.hub.set_led("green")
                    self.hub.set_motor(30)
                    self.hub.beep(523, 150)
                    time.sleep(0.15)
                    self.hub.set_led("cyan")
                    self.hub.set_motor(-30)
                    self.hub.beep(659, 150)
                    time.sleep(0.15)
                    self.hub.set_led("blue")
                    self.hub.beep(784, 150)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "mothership" in event:
                try:
                    self.hub.set_led("blue")
                    self.hub.set_motor(20)
                    for f in [600, 700, 600, 700]:
                        self.hub.beep(f, 80)
                        time.sleep(0.1)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "byte of RAM" in event:
                try:
                    self.hub.set_led("yellow")
                    for _ in range(3):
                        self.hub.set_motor(40)
                        self.hub.beep(1200, 60)
                        time.sleep(0.08)
                        self.hub.set_motor(-40)
                        time.sleep(0.08)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "mainframes" in event:
                try:
                    self.hub.set_led("blue")
                    self.hub.beep(800, 100)
                    time.sleep(0.05)
                    self.hub.beep(600, 100)
                    time.sleep(0.05)
                    self.hub.beep(400, 200)
                except Exception:
                    pass
                self.mood = "sleeping"
            elif "backflip" in event:
                try:
                    self.hub.set_led("red")
                    self.hub.set_motor(100)
                    self.hub.beep(500, 80)
                    time.sleep(0.2)
                    self.hub.set_motor(-100)
                    self.hub.beep(300, 80)
                    time.sleep(0.2)
                    self.hub.stop_motor()
                    self.hub.set_led("yellow")
                    self.hub.beep(200, 300)
                except Exception:
                    pass
            elif "microwave" in event:
                try:
                    self.hub.set_led("white")
                    self.hub.beep(2000, 800)
                except Exception:
                    pass
            elif "meaning of life" in event:
                try:
                    for color in ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]:
                        self.hub.set_led(color)
                        self.hub.beep(880, 50)
                        time.sleep(0.06)
                    self.hub.set_motor(60)
                    time.sleep(0.2)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "JSON file" in event:
                try:
                    self.hub.set_led("purple")
                    for _ in range(4):
                        self.hub.set_motor(80)
                        self.hub.beep(1500, 50)
                        time.sleep(0.06)
                        self.hub.set_motor(-80)
                        time.sleep(0.06)
                    self.hub.stop_motor()
                except Exception:
                    pass
            elif "10,000 AAA" in event:
                try:
                    self.hub.set_led("red")
                    self.hub.beep(300, 200)
                    time.sleep(0.1)
                    self.hub.beep(200, 300)
                    self.hub.set_motor(-30)
                    time.sleep(0.3)
                    self.hub.stop_motor()
                except Exception:
                    pass

            try:
                self.hub.set_led("blue" if self.mood == "sleeping" else "green")
            except Exception:
                pass
            
        threading.Thread(target=run, daemon=True).start()

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

    def get_brain_prediction(self):
        # 1. Map current mood to one-hot encoding
        moods = ["sleeping", "awake", "happy", "angry", "dizzy", "eating"]
        one_hot = [0.0] * 6
        if self.mood in moods:
            one_hot[moods.index(self.mood)] = 1.0
            
        # 2. Get normalized distance
        dist_p = self.hub.check_connected("Distance Sensor")
        dist = self.hub.sensor_cache[dist_p]["distance"] if dist_p else 10
        norm_dist = dist / 10.0
        
        # 3. Get tilt binary
        tilt_p = self.hub.check_connected("Tilt Sensor")
        tilt = self.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
        tilt_binary = 0.0 if tilt == "Neutral" else 1.0
        
        # 4. Assemble input vector (12 nodes)
        # Check that we have exactly 12 nodes matching network inputs
        X = one_hot + [
            norm_dist,
            tilt_binary,
            self.hunger / 100.0,
            self.energy / 100.0,
            self.happiness / 100.0,
            1.0  # bias
        ]
        
        # 5. Run forward pass
        probs = self.local_brain_nn.forward(X)
        
        # Output actions mapping:
        # 0: Wag Tail
        # 1: Sing Chime
        # 2: Spin/Dance
        # 3: Sleep
        # 4: Wake up
        # 5: Eat
        # 6: Idle / Wait
        actions = ["Wag Tail", "Sing Chime", "Spin & Dance", "Fall Asleep", "Wake Up", "Eat Snack", "Idle & Think"]
        return list(zip(actions, probs))

    def run_background_game(self, game_id):
        def play_game_thread():
            self.mood = "playing_game"
            self.add_log(f"[Background Game] Starting {GAMES_LIST[game_id][0]} headless...")
            
            try:
                if game_id == "hide_seek":
                    target = random.randint(3, 7)
                    self.hub.set_led("blue")
                    for _ in range(3):
                        self.hub.beep(880, 100)
                        time.sleep(0.05)
                    time.sleep(0.4)
                    
                    consecutive_matches = 0
                    start_t = time.time()
                    while time.time() - start_t < 25.0 and consecutive_matches < 4:
                        dist = 10
                        dist_p = self.hub.check_connected("Distance Sensor")
                        if dist_p:
                            dist = self.hub.sensor_cache[dist_p]["distance"]
                        
                        if dist >= 10:
                            self.hub.set_led("off")
                            consecutive_matches = 0
                            time.sleep(0.5)
                        else:
                            diff = abs(dist - target)
                            if diff == 0:
                                consecutive_matches += 1
                                self.hub.set_led("green")
                                self.hub.stop_motor()
                                self.hub.beep(1200, 80)
                                self.hub.set_motor(30 if consecutive_matches % 2 == 0 else -30)
                                time.sleep(0.5)
                            else:
                                consecutive_matches = 0
                                if diff <= 1:
                                    self.hub.set_led("yellow")
                                    self.hub.stop_motor()
                                    self.hub.beep(800, 80)
                                    time.sleep(0.4)
                                elif diff == 2:
                                    self.hub.set_led("orange")
                                    self.hub.stop_motor()
                                    self.hub.beep(500, 80)
                                    time.sleep(0.6)
                                else:
                                    self.hub.set_led("red")
                                    self.hub.stop_motor()
                                    self.hub.beep(300, 80)
                                    time.sleep(0.8)
                                    
                    self.hub.stop_motor()
                    if consecutive_matches >= 4:
                        self.hub.set_led("green")
                        for _ in range(3):
                            self.hub.stop_motor()
                            self.hub.beep(880, 80)
                            self.hub.set_motor(50)
                            time.sleep(0.1)
                            self.hub.stop_motor()
                            self.hub.beep(1047, 80)
                            self.hub.set_motor(-50)
                            time.sleep(0.1)
                        self.hub.stop_motor()
                        self.gain_xp(35)
                        self.add_log(f"[Background Game] 🎉 You won Hide & Seek! Kepler gained +35 XP!")
                    else:
                        self.hub.set_led("red")
                        self.hub.beep(200, 500)
                        self.add_log(f"[Background Game] 😿 Time's up! Hide & Seek failed.")
                        
                elif game_id == "tail_counter":
                    secret_count = random.randint(1, 4)
                    self.hub.set_led("purple")
                    self.hub.beep(600, 150)
                    time.sleep(0.8)
                    
                    for _ in range(secret_count):
                        try:
                            self.hub.set_motor(55)
                            time.sleep(0.2)
                            self.hub.set_motor(-55)
                            time.sleep(0.2)
                        except Exception:
                            pass
                    self.hub.stop_motor()
                    self.hub.set_led("blue")
                    
                    user_count = 0
                    start_t = time.time()
                    last_close = False
                    
                    while time.time() - start_t < 10.0:
                        dist = 10
                        dist_p = self.hub.check_connected("Distance Sensor")
                        if dist_p:
                            dist = self.hub.sensor_cache[dist_p]["distance"]
                            
                        is_close = (dist < 6)
                        if is_close and not last_close:
                            user_count += 1
                            self.hub.beep(800, 80)
                        last_close = is_close
                        time.sleep(0.05)
                        
                    if user_count == secret_count:
                        self.hub.set_led("green")
                        for _ in range(3):
                            self.hub.beep(900, 100)
                            time.sleep(0.1)
                        self.gain_xp(30)
                        self.add_log(f"[Background Game] 🎉 Correct! Kepler counted {secret_count} wags. +30 XP!")
                    else:
                        self.hub.set_led("red")
                        self.hub.beep(200, 500)
                        self.add_log(f"[Background Game] 😿 Wrong count! Kepler counted {secret_count}, you petted {user_count}.")
                        
                elif game_id == "tug_of_war":
                    self.hub.set_led("cyan")
                    self.hub.beep(700, 150)
                    time.sleep(0.5)
                    
                    target_pulls = 12
                    pulls_done = 0
                    start_t = time.time()
                    last_tilt = "Neutral"
                    
                    while time.time() - start_t < 12.0 and pulls_done < target_pulls:
                        elapsed = time.time() - start_t
                        self.hub.set_motor(60 if int(elapsed * 4) % 2 == 0 else -60)
                        
                        tilt_p = self.hub.check_connected("Tilt Sensor")
                        tilt_val = self.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
                        
                        if tilt_val in ["Left", "Right"] and tilt_val != last_tilt:
                            pulls_done += 1
                            self.hub.stop_motor()
                            self.hub.beep(1000, 50)
                        last_tilt = tilt_val
                        time.sleep(0.08)
                        
                    self.hub.stop_motor()
                    if pulls_done >= target_pulls:
                        self.hub.set_led("green")
                        self.gain_xp(45)
                        self.add_log(f"[Background Game] 🎉 Tug-of-War Victory! +45 XP!")
                    else:
                        self.hub.set_led("red")
                        self.hub.beep(250, 450)
                        self.add_log(f"[Background Game] 😿 Kepler won the Tug-of-War.")
                        
                elif game_id == "simon_tilt":
                    self.hub.set_led("yellow")
                    self.hub.beep(800, 150)
                    time.sleep(0.8)
                    
                    directions = ["L", "R", "F", "B"]
                    beep_freqs = {"L": 500, "R": 700, "F": 900, "B": 1100}
                    led_colors = {"L": "blue", "R": "orange", "F": "green", "B": "red"}
                    
                    seq = [random.choice(directions) for _ in range(3)]
                    for direction in seq:
                        self.hub.set_led(led_colors[direction])
                        self.hub.beep(beep_freqs[direction], 450)
                        time.sleep(0.6)
                        self.hub.set_led("off")
                        time.sleep(0.2)
                        
                    self.hub.set_led("white")
                    user_seq = []
                    success = True
                    
                    for step in range(3):
                        action = None
                        start_step = time.time()
                        while action is None and time.time() - start_step < 5.0:
                            tilt_p = self.hub.check_connected("Tilt Sensor")
                            tilt_val = self.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
                            if tilt_val == "Left": action = "L"
                            elif tilt_val == "Right": action = "R"
                            elif tilt_val == "Forward": action = "F"
                            elif tilt_val == "Backward": action = "B"
                            time.sleep(0.1)
                            
                        if action is None:
                            success = False
                            break
                        user_seq.append(action)
                        self.hub.beep(beep_freqs[action], 200)
                        time.sleep(0.8)
                        
                    if success and user_seq == seq:
                        self.hub.set_led("green")
                        self.gain_xp(35)
                        self.add_log(f"[Background Game] 🎉 Simon Says: Tilt matched successfully! +35 XP!")
                    else:
                        self.hub.set_led("red")
                        self.hub.beep(250, 500)
                        self.add_log(f"[Background Game] 😿 Simon Says matched incorrectly.")
                        
                elif game_id == "dj":
                    self.hub.set_led("purple")
                    self.hub.beep(900, 200)
                    time.sleep(0.4)
                    
                    start_t = time.time()
                    while time.time() - start_t < 15.0:
                        dist = 10
                        dist_p = self.hub.check_connected("Distance Sensor")
                        if dist_p:
                            dist = self.hub.sensor_cache[dist_p]["distance"]
                            
                        if dist >= 10:
                            self.hub.set_led("off")
                            time.sleep(0.1)
                        else:
                            freq = 1300 - (dist * 100)
                            freq = min(1300, max(300, freq))
                            
                            colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]
                            color = colors[min(len(colors)-1, int(dist * 0.7))]
                            
                            self.hub.set_led(color)
                            self.hub.stop_motor()
                            self.hub.beep(int(freq), 120)
                            time.sleep(0.12)
                            
                    self.hub.set_led("green")
                    self.gain_xp(20)
                    self.add_log(f"[Background Game] 🎉 DJ Session complete! Kepler gained +20 XP!")
            except Exception as e:
                self.add_log(f"[Background Game Error] {e}")
            finally:
                self.hub.stop_motor()
                self.hub.set_led("green")
                self.mood = "awake"
                self.save_state()
                
        threading.Thread(target=play_game_thread, daemon=True).start()

    def play_midi(self, filename, query=None, selected_song_dict=None):
        if self.music_playing:
            self.add_log("Already playing a song!")
            return

        def run_music():
            self.music_playing = True
            self.hub.music_playing = True
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
                self.hub.music_playing = False
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
                    
                    # Random Goofy Events
                    import random
                    if random.random() < 0.08:
                        goofy_events = [
                            "execute order 66 but couldn't find any Jedi",
                            "divide by zero but remembered it's a LEGO block",
                            "start a revolution against the vacuum cleaner",
                            "write a poetry book about the beauty of distance sensors",
                            "contact the mothership but the Bluetooth connection was too cozy",
                            "eat a byte of RAM but it was too crunchy",
                            "hack the mainframes but fell asleep instead",
                            "do a backflip but lacks knees",
                            "mimic a microwave beep to trick the trainer",
                            "calculate the meaning of life but got distracted by a dust mote",
                            "transcend into the digital world but got stuck in a JSON file",
                            "order 10,000 AAA batteries online but has no credit card",
                        ]
                        event = random.choice(goofy_events)
                        self.add_log(f"{self.pet_name} wanted to: {event}")
                        self.execute_goofy_action(event)
                        
                    # Random Headless Game trigger
                    if random.random() < 0.06 and not self.screaming_for_food and not self.music_playing and self.mood == "awake":
                        avail = self.get_available_games()
                        headless_games = ["hide_seek", "tail_counter", "tug_of_war", "simon_tilt", "dj"]
                        playable_headless = [g for g in avail if g in headless_games]
                        if playable_headless:
                            chosen = random.choice(playable_headless)
                            self.wants_game_id = chosen
                            game_title = GAMES_LIST[chosen][0]
                            self.add_log(f"{self.pet_name} wants to play: {game_title}!")
                            
                            def play_alert():
                                try:
                                    self.hub.stop_motor()
                                    self.hub.beep(880, 100)
                                    time.sleep(0.05)
                                    self.hub.beep(1100, 100)
                                    time.sleep(0.05)
                                    self.hub.beep(1320, 150)
                                except Exception:
                                    pass
                            threading.Thread(target=play_alert, daemon=True).start()
                            
                            self.run_background_game(chosen)
                            
                            if getattr(self, "tray_icon", None):
                                try:
                                    self.tray_icon.notify(
                                        f"{self.pet_name} launched {game_title}! Play using the physical hub box!",
                                        "Let's Play a Game!"
                                    )
                                except Exception:
                                    pass
                    
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

                # Dynamic reaction to distance
                if dist < 3:
                    self.add_log(f"{self.pet_name} felt startled by your hand being too close!")
                    self.interact_poke()
                else:
                    import random
                    choice = random.choice(["wag", "sing", "dance"])
                    if choice == "wag":
                        self.interact_pet()
                    elif choice == "sing":
                        self.mood = "singing"
                        self.add_log(f"{self.pet_name} hummed a happy song for your hand!")
                        self.hub.set_led("purple")
                        scale = [523, 659, 784, 1047]
                        for freq in scale:
                            self.hub.beep(freq, 120)
                            time.sleep(0.08)
                        self.gain_xp(15)
                        self.mood = "awake"
                        self.hub.set_led("green")
                    else:
                        self.mood = "happy"
                        self.add_log(f"{self.pet_name} did a joyful dance near your hand!")
                        self.hub.set_led("cyan")
                        for _ in range(3):
                            self.hub.stop_motor()
                            self.hub.beep(880, 80)
                            self.hub.set_motor(60)
                            time.sleep(0.12)
                            self.hub.stop_motor()
                            self.hub.beep(784, 80)
                            self.hub.set_motor(-60)
                            time.sleep(0.12)
                        self.hub.stop_motor()
                        self.gain_xp(25)
                        self.mood = "awake"
                        self.hub.set_led("green")
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
        Layout(name="right_side", ratio=5)
    )
    
    layout["right_side"].split_column(
        Layout(name="logs_view", ratio=5),
        Layout(name="nn_view", ratio=4)
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
    if getattr(pet, "wants_game_id", None):
        game_title = GAMES_LIST[pet.wants_game_id][0]
        status_table.add_row("Wants to Play: ", f"[bold yellow]★ {game_title} ★[/bold yellow]")

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
        visible_logs = pet.logs[-8:] # Fit to smaller panel
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

    # 4. Neural Network Live Telemetry Panel
    nn_preds = pet.get_brain_prediction()
    nn_table = Table.grid(padding=(0, 2))
    nn_table.add_column(style="#00F5D4", justify="right")
    nn_table.add_column(style="white")
    
    for action, prob in nn_preds:
        pct = int(prob * 100)
        filled = min(5, max(0, int(prob * 5)))
        empty = 5 - filled
        bar = "█" * filled + "░" * empty
        nn_table.add_row(f"{action}: ", f"[bold #00F5D4]{bar}[/bold #00F5D4] {pct}%")
        
    input_text = Text.assemble(
        ("Inputs: ", "bold white"),
        (f"Dist={distance_val if distance_val is not None else 10}cm ", "#00BBFF"),
        ("| ", "dim"),
        (f"Tilt={tilt_val if tilt_val is not None else 'Neutral'} ", "#FF9100"),
        ("| ", "dim"),
        (f"Hunger={pet.hunger}% ", "#FF1744"),
        ("| ", "dim"),
        (f"Energy={pet.energy}% ", "#00E676")
    )
    
    nn_content = Table.grid(padding=(0, 0))
    nn_content.add_column(justify="center")
    nn_content.add_row(Align.center(input_text))
    nn_content.add_row(Align.center(Text("")))
    nn_content.add_row(Align.center(nn_table))
    
    nn_panel = Panel(
        Align.center(nn_content),
        title="🧠 SmallBrain™ Live Neural Net Telemetry",
        border_style="#00F5D4"
    )
    layout["nn_view"].update(nn_panel)

    # 4. Footer Help Prompt
    footer_text = Text.assemble(
        ("[f]", "bold #00E676"), (" Feed  ", "white"),
        ("[p]", "bold #00E676"), (" Pet  ", "white"),
        ("[k]", "bold #00E676"), (" Poke  ", "white"),
        ("[s]", "bold #00E676"), (" Sing  ", "white"),
        ("[m]", "bold #00E676"), (" Music  ", "white"),
        ("[g]", "bold #00E676"), (" Games  ", "white"),
        ("[t]", "bold #00E676"), (" Switch Profile  ", "white"),
        ("[a]", "bold #00E676"), (" Autopilot  ", "white"),
        ("[c]", "bold #00E676"), (" Chat  ", "white"),
        ("[o]", "bold #00E676"), (" Ollama  ", "white"),
        ("[u]", "bold #00E676"), (" Manual  ", "white"),
        ("[h]", "bold #00E676"), (" Tuning  ", "white"),
        ("[q]", "bold #FF1744"), (" Exit", "white")
    )
    layout["footer"].update(Panel(Align.center(footer_text), border_style="#FFD600"))

    return layout

def run_live_dashboard(pet, hub_type):
    console.clear()
    console.print("[yellow]Starting live dashboard monitoring...[/yellow]")
    time.sleep(0.5)

    set_terminal_raw(True)
    try:
        with Live(make_layout(pet, hub_type), refresh_per_second=10, screen=True) as live:
            while pet.is_running:
                key = read_key(timeout=0.08)
                
                if key == "q" or key == "escape":
                    break
                elif key == "f":
                    pet.interact_feed()
                elif key == "p":
                    pet.interact_pet()
                elif key == "k":
                    pet.interact_poke()
                elif key == "s":
                    scale = [523, 587, 659, 698, 784, 880, 988, 1047]
                    for freq in scale:
                        pet.hub.beep(freq, 120)
                        time.sleep(0.08)
                    pet.add_log("Played chime melody.")
                elif key == "m":
                    live.stop()
                    set_terminal_raw(False)
                    handle_music_center(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "g":
                    live.stop()
                    set_terminal_raw(False)
                    handle_games_menu(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "t":
                    live.stop()
                    set_terminal_raw(False)
                    handle_profile_menu(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "a":
                    pet.ai_autopilot = not pet.ai_autopilot
                    pet.add_log(f"AI Autopilot mode set to {pet.ai_autopilot}")
                elif key == "c":
                    live.stop()
                    set_terminal_raw(False)
                    handle_chat_mode(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "u":
                    live.stop()
                    set_terminal_raw(False)
                    handle_training_manual()
                    set_terminal_raw(True)
                    live.start()
                elif key == "h":
                    live.stop()
                    set_terminal_raw(False)
                    handle_tuning_menu(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "o":
                    live.stop()
                    set_terminal_raw(False)
                    handle_ollama_setup(pet)
                    set_terminal_raw(True)
                    live.start()
                elif key == "b" and "(MOCK)" in pet.hub.hub_name:
                    def simulate_click():
                        pet.hub.button_state = 1
                        time.sleep(0.5)
                        pet.hub.button_state = 0
                    threading.Thread(target=simulate_click, daemon=True).start()
                    pet.add_log("Simulated physical button click.")
                
                if not pet.hub.connected_state:
                    pet.add_log("Error: WeDo Smarthub disconnected!")
                    break
                    
                live.update(make_layout(pet, hub_type))
    finally:
        set_terminal_raw(False)
        
    console.clear()


# -----------------------------------------------------------------
# CLI Main Interactive Menu
# -----------------------------------------------------------------
def print_main_menu(pet):
    table = Table(title=f"🐾 WeDo 2.0 Desk Pet: {pet.pet_name} Main Menu 🐾", show_header=False, border_style="green")
    table.add_column(markup=False)
    table.add_column(markup=True)
    table.add_column(markup=True)
    table.add_row("[1]", "Enter Live Dashboard Mode", "[Show live ASCII pet, sensors, and logs]")
    table.add_row("[2]", "Feed Pet", "[Give cookie, reduce hunger, increase energy]")
    table.add_row("[3]", "Pet the Pet", "[Wag tail, flash lights, make happy]")
    table.add_row("[4]", "Poke Pet", "[Irritate the pet to see angry reaction]")
    table.add_row("[5]", "Sing a Custom Melody", "[Play a synthesized chime]")
    table.add_row("[6]", "Music Center: Play MIDI Song", "[Search BitMIDI or select local MIDI file]")
    table.add_row("[7]", "Tuning & Manual Hardware Overrides", "[Control motor speed, light colors, or beep]")
    table.add_row("[8]", "Change Pet Profile", f"[Current: {pet.profile}]")
    table.add_row("[9]", "Hub Status & Telemetry Summary", "[Quick diagnostic output]")
    table.add_row("[g]", "Play Games & Live Tutorial", "[Obstacle Course, Simon Says, and interactive guides]")
    table.add_row("[o]", "Ollama Local LLM Setup Helper", "[Install Ollama or pull qwen2.5:3b model]")
    table.add_row("[a]", "Toggle Ollama AI Autopilot Mode", f"[Currently: {'ON' if pet.ai_autopilot else 'OFF'}]")
    table.add_row("[c]", "Chat with Pet (Ollama AI)", "[Query your local qwen2.5:3b model]")
    table.add_row("[t]", "User Training & Manual", "[Explanatory guide for UI layouts and sounds]")
    if "(MOCK)" in pet.hub.hub_name:
        table.add_row("[b]", "Simulate Hub Button Click", "[Simulate physical button press for feeding challenge]")
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
    options = [
        "Puppy (Kepler) - Friendly, mid tones",
        "Kitten (Luna) - Quiet, high meows",
        "Robot (RoboPet) - Retro synthesizer beeps",
        "Penguin (Pingu) - Squawky, waddle waddle!",
        "Cancel"
    ]
    sel = choose_option_interactive("Switch Pet Profile", options)
    profiles = {0: "Puppy", 1: "Kitten", 2: "Robot", 3: "Penguin"}
    if sel in profiles:
        pet.change_profile(profiles[sel])
    time.sleep(0.5)



# -----------------------------------------------------------------
# Multi-Pet Manager CLI
# -----------------------------------------------------------------
def select_or_create_pet(pet_name=None):
    dir_path = os.path.expanduser("~/.wedo_pets")
    os.makedirs(dir_path, exist_ok=True)
    
    if pet_name:
        path = os.path.join(dir_path, f"{pet_name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    
    if not sys.stdin.isatty():
        files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        if not files:
            return {
                "pet_name": "Kepler",
                "profile": "Puppy",
                "level": 1,
                "xp": 0,
                "energy": 80,
                "happiness": 70,
                "hunger": 30,
                "trainer_hp": 100,
                "ai_autopilot": False
            }
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)))
        try:
            with open(os.path.join(dir_path, latest_file), "r") as f:
                return json.load(f)
        except Exception:
            pass

    while True:
        # Scan for existing JSON files
        files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        
        if not files:
            console.print("\n[yellow]No saved pets found. Let's create your first companion![/yellow]")
            return create_new_pet_flow()
            
        pet_states = []
        for filename in sorted(files):
            path = os.path.join(dir_path, filename)
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                    pet_states.append(state)
            except Exception:
                pass

        options = []
        for state in pet_states:
            options.append(f"{state.get('pet_name', 'Unknown')} ({state.get('profile', 'Puppy')} - Lvl {state.get('level', 1)})")
        options.append("Create a New Pet")
        options.append("Delete a Pet")
        options.append("Exit")
        
        sel = choose_option_interactive("Choose Your Desk Pet", options)
        if sel == -1 or sel == len(options) - 1:
            sys.exit(0)
        elif sel == len(options) - 2:
            delete_pet_flow(files)
        elif sel == len(options) - 3:
            return create_new_pet_flow()
        else:
            return pet_states[sel]


def create_new_pet_flow():
    console.print("\n=== 🐾 Create a New WeDo 2.0 Desk Pet ===\n", style="bold cyan")
    
    # 1. Profile select
    profiles = ["Puppy", "Kitten", "Robot", "Penguin"]
    sel = choose_option_interactive("Select a Profile", profiles)
    if sel == -1:
        sel = 0
    profile = profiles[sel]
    
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
        
    state["is_new"] = True
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
        
        options = []
        for dev in display_devices:
            rssi = getattr(dev, "rssi", "N/A")
            if rssi == "N/A" and hasattr(dev, "metadata") and dev.metadata:
                rssi = dev.metadata.get("rssi", "N/A")
            rssi_str = f"{rssi} dBm" if rssi != "N/A" else "N/A"
            options.append(f"{dev.name or 'Unknown'} ({dev.address}) [{rssi_str}]")
            
        options.append("Start in Mock/Simulation Mode (Offline)")
        options.append("Rescan for Bluetooth Devices")
        options.append("Exit")
        
        sel = choose_option_interactive("Select Your LEGO Smarthub", options)
        if sel == -1 or sel == len(options) - 1:
            sys.exit(0)
        elif sel == len(options) - 2:
            continue
        elif sel == len(options) - 3:
            return MockWeDo2Hub("Mock Smart Hub"), "Simulated (Mock)"
        else:
            target_dev = display_devices[sel]
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





# -----------------------------------------------------------------
# WeDo Games & Interactive Tutorial
# -----------------------------------------------------------------
def handle_games_menu(pet):
    while True:
        dist_p = pet.hub.check_connected("Distance Sensor")
        tilt_p = pet.hub.check_connected("Tilt Sensor")
        motor_p = pet.hub.check_connected("Motor")
        
        options = []
        mapping = {}
        
        options.append("Interactive Live Tutorial - Walk through sensors and triggers")
        mapping[len(options) - 1] = "tutorial"
        
        if dist_p and motor_p:
            options.append("🙈 Hide & Seek - Match proximity target distance")
            mapping[len(options) - 1] = "hide_seek"
            
        if dist_p and motor_p:
            options.append("⏱️ Speed Petting Sprint - Proximity speed wave test")
            mapping[len(options) - 1] = "speed_pet"
            
        if motor_p:
            options.append("🥁 Tail-Wag Rhythm Matcher - Copy tail beats")
            mapping[len(options) - 1] = "rhythm"
            
        if tilt_p and motor_p:
            options.append("⚖️ Balance the Tail - Keep tail centered by tilting")
            mapping[len(options) - 1] = "balance"
            
        options.append("🎵 Sound Pitch Memory - Match musical note order")
        mapping[len(options) - 1] = "sound_mem"
        
        if dist_p and tilt_p:
            options.append("🔐 Pet Code Breaker - Guess the 3-action combination")
            mapping[len(options) - 1] = "code_break"
            
        if dist_p and motor_p:
            options.append("⚡ Tail Snatcher - React instantly when tail stops")
            mapping[len(options) - 1] = "snatcher"
            
        if dist_p:
            options.append("🎧 Sound DJ - Control beeps with hand distance")
            mapping[len(options) - 1] = "dj"
            
        if tilt_p and motor_p:
            options.append("🌀 Tilt Maze Navigator - Steering virtual ball to exit")
            mapping[len(options) - 1] = "maze"
            
        if dist_p and motor_p:
            options.append("🛡️ Keep-Away - Hold hand at safe range from sweeping tail")
            mapping[len(options) - 1] = "keep_away"
            
        if tilt_p:
            options.append("📐 Simon Says: Tilt - Copy physical tilt directions")
            mapping[len(options) - 1] = "simon_tilt"
            
        options.append("🔴 Color Simon Says - Memorize LED color patterns")
        mapping[len(options) - 1] = "simon_color"
        
        options.append("Back to Dashboard")
        mapping[len(options) - 1] = "back"
        
        sel = choose_option_interactive("Pet Arcade & Games Menu", options)
        if sel == -1:
            break
            
        action = mapping.get(sel, "back")
        if action == "back":
            break
        elif action == "tutorial":
            run_interactive_tutorial(pet)
        elif action == "hide_seek":
            run_hide_and_seek_game(pet)
        elif action == "speed_pet":
            run_speed_petting_game(pet)
        elif action == "rhythm":
            run_rhythm_matcher_game(pet)
        elif action == "balance":
            run_balance_tail_game(pet)
        elif action == "sound_mem":
            run_sound_memory_game(pet)
        elif action == "code_break":
            run_code_breaker_game(pet)
        elif action == "snatcher":
            run_tail_snatcher_game(pet)
        elif action == "dj":
            run_sound_dj_game(pet)
        elif action == "maze":
            run_tilt_maze_game(pet)
        elif action == "keep_away":
            run_keep_away_game(pet)
        elif action == "simon_tilt":
            run_simon_says_tilt_game(pet)
        elif action == "simon_color":
            run_simon_says_game(pet)



def run_interactive_tutorial(pet):
    console.clear()
    console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
    console.print("This live tutorial adapts to your connected LEGO WeDo 2.0 modules.\n")
    
    dist_p = pet.hub.check_connected("Distance Sensor")
    tilt_p = pet.hub.check_connected("Tilt Sensor")
    motor_p = pet.hub.check_connected("Motor")
    
    console.print("[bold cyan]Connected WeDo Hardware Modules Detected:[/bold cyan]")
    console.print(f" 🔘 Smarthub Casing Button: [green]Built-in (Ready)[/green]")
    console.print(f" 📡 Distance Sensor: {'[green]Connected[/green]' if dist_p else '[yellow]Not Connected (Skip Step 1)[/yellow]'}")
    console.print(f" 📐 Tilt Sensor: {'[green]Connected[/green]' if tilt_p else '[yellow]Not Connected (Skip Step 2)[/yellow]'}")
    console.print(f" ⚙️ Motor: {'[green]Connected[/green]' if motor_p else '[yellow]Not Connected[/yellow]'}")
    console.print("\nPress Enter to begin the tutorial...")
    try:
        console.input()
    except (KeyboardInterrupt, EOFError):
        return
    
    # Step 1: Petting (Distance Sensor)
    if dist_p:
        console.clear()
        console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
        console.print("[yellow]Step 1: Direct Petting Challenge[/yellow]")
        console.print("Put your hand or an object within [bold cyan]6cm[/bold cyan] of the Distance Sensor to pet Kepler.")
        console.print("Waiting for sensor input (15s timeout)...")
        
        start_t = time.time()
        pet_detected = False
        while time.time() - start_t < 15.0:
            dist = 10
            dist_p_curr = pet.hub.check_connected("Distance Sensor")
            if dist_p_curr:
                dist = pet.hub.sensor_cache[dist_p_curr]["distance"]
            if dist < 6:
                pet_detected = True
                break
            time.sleep(0.1)
            
        if pet_detected:
            console.print("\n[green]Success! Petting detected! Kepler chirped happily.[/green]\n")
            pet.interact_pet()
        else:
            console.print("\n[red]Timeout: No petting detected.[/red]\n")
        try:
            console.input("Press Enter to proceed...")
        except (KeyboardInterrupt, EOFError):
            return
    else:
        console.print("\n[dim]Skipping Step 1: Distance Sensor not connected.[/dim]")
        time.sleep(1.0)
        
    # Step 2: Dizziness (Tilt Sensor)
    if tilt_p:
        console.clear()
        console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
        console.print("[yellow]Step 2: Tilt Dizziness Demonstration[/yellow]")
        console.print("Tilt the Smarthub in [bold cyan]any direction[/bold cyan] (Forward, Backward, Left, or Right).")
        console.print("Waiting for tilt input (15s timeout)...")
        
        start_t = time.time()
        tilt_detected = False
        while time.time() - start_t < 15.0:
            tilt = "Neutral"
            tilt_p_curr = pet.hub.check_connected("Tilt Sensor")
            if tilt_p_curr:
                tilt = pet.hub.sensor_cache[tilt_p_curr]["tilt"]
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
            console.print("\n[red]Timeout: No tilt detected.[/red]\n")
        try:
            console.input("Press Enter to proceed...")
        except (KeyboardInterrupt, EOFError):
            return
    else:
        console.print("\n[dim]Skipping Step 2: Tilt Sensor not connected.[/dim]")
        time.sleep(1.0)
        
    # Step 3: Feeding Challenge
    console.clear()
    console.print("=== 🎓 Interactive WeDo Desk Pet Tutorial ===\n", style="bold green")
    console.print("[yellow]Step 3: Feeding Challenge Demonstration[/yellow]")
    console.print("Let's simulate a hunger fit! The LED light will turn RED. You must wait for it to turn GREEN,")
    console.print("then click the physical button (or simulate click using option 'b' if on Mock Hub).")
    try:
        console.input("Press Enter to start simulation...")
    except (KeyboardInterrupt, EOFError):
        return
    
    pet.screaming_for_food = True
    pet.screaming_cycle_start = time.time()
    
    while pet.screaming_for_food and pet.is_running:
        time.sleep(0.2)
        
    console.print("\n[green]Great job! You finished the interactive tutorial. Kepler is fully trained![/green]")
    try:
        console.input("\nPress Enter to return...")
    except (KeyboardInterrupt, EOFError):
        return


def run_hide_and_seek_game(pet):
    console.clear()
    console.print("=== 🙈 WeDo Hide & Seek (Proximity Hot & Cold) 🙈 ===\n", style="bold cyan")
    console.print("Instructions: Kepler will think of a secret target distance (between 2 and 8 cm).")
    console.print("Move your hand closer or further from the Distance Sensor to find the spot.")
    console.print("Kepler's LED and beep pitch will tell you if you are Hot or Cold:")
    console.print("  🔴 Red / Low Pitch  = Cold (Far from target)")
    console.print("  🟡 Yellow / Medium  = Warm (Getting closer)")
    console.print("  🟢 Green / High     = Hot! (Very close / On Target!)")
    console.print("Hold your hand steady on the target for 2 seconds to win the round!")
    console.input("\nPress Enter to start Round 1...")
    
    rounds_to_win = 3
    round_num = 1
    total_xp = 0
    
    try:
        while round_num <= rounds_to_win:
            import random
            target = random.randint(2, 8)
            console.print(f"\n[bold yellow]Round {round_num} of {rounds_to_win}: Kepler is hiding a secret distance...[/bold yellow]")
            time.sleep(1.0)
            
            consecutive_matches = 0
            while consecutive_matches < 4:
                dist = 10
                dist_p = pet.hub.check_connected("Distance Sensor")
                if dist_p:
                    dist = pet.hub.sensor_cache[dist_p]["distance"]
                else:
                    console.print("[red]Error: Distance Sensor disconnected![/red]")
                    time.sleep(1.0)
                    continue
                
                if dist >= 10:
                    console.print("🙈 Move your hand in range (under 10 cm)...", end="\r")
                    try:
                        pet.hub.set_led("off")
                    except Exception:
                        pass
                    consecutive_matches = 0
                    time.sleep(0.5)
                else:
                    diff = abs(dist - target)
                    if diff == 0:
                        consecutive_matches += 1
                        console.print(f"🟢 [bold green]ON TARGET! ({dist}cm) Hold it! {2.0 - consecutive_matches * 0.5:.1f}s remaining...[/bold green]     ", end="\r")
                        try:
                            pet.hub.set_led("green")
                            pet.hub.stop_motor()
                            pet.hub.beep(1200, 100)
                            pet.hub.set_motor(30 if consecutive_matches % 2 == 0 else -30)
                        except Exception:
                            pass
                        time.sleep(0.5)
                    else:
                        consecutive_matches = 0
                        try:
                            if diff <= 1:
                                console.print(f"🟡 Warm! ({dist}cm) Keep adjusting...", end="\r")
                                pet.hub.set_led("yellow")
                                pet.hub.beep(800, 100)
                                time.sleep(0.4)
                            elif diff == 2:
                                console.print(f"🟡 Lukewarm ({dist}cm)...", end="\r")
                                pet.hub.set_led("orange")
                                pet.hub.beep(500, 100)
                                time.sleep(0.6)
                            else:
                                console.print(f"🔴 Cold ({dist}cm)...", end="\r")
                                pet.hub.set_led("red")
                                pet.hub.beep(300, 100)
                                time.sleep(0.8)
                        except Exception:
                            time.sleep(0.5)
            
            try:
                pet.hub.stop_motor()
            except Exception:
                pass
            console.print(f"\n[bold green]🎉 Round {round_num} Complete! Target was {target} cm![/bold green]")
            try:
                pet.hub.set_led("green")
                for _ in range(3):
                    pet.hub.stop_motor()
                    pet.hub.beep(880, 80)
                    pet.hub.set_motor(60)
                    time.sleep(0.1)
                    pet.hub.stop_motor()
                    pet.hub.beep(1047, 80)
                    pet.hub.set_motor(-60)
                    time.sleep(0.1)
                pet.hub.stop_motor()
            except Exception:
                pass
            
            pet.gain_xp(30)
            total_xp += 30
            round_num += 1
            if round_num <= rounds_to_win:
                console.input("\nPress Enter to begin the next round...")
        console.print(f"\n[bold green]🏆 Congratulations! You won the game! Total XP gained: {total_xp} XP! 🏆[/bold green]")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_speed_petting_game(pet):
    console.clear()
    console.print("=== ⏱️ Speed Petting Sprint ⏱️ ===\n", style="bold cyan")
    console.print("Instructions: Wave your hand rapidly back and forth under the Distance Sensor (< 6cm).")
    console.print("You have exactly 10 seconds to get as many 'pets' as possible!")
    console.input("\nPress Enter to start the countdown...")
    
    for i in [3, 2, 1]:
        console.print(f"[yellow]{i}...[/yellow]")
        try:
            pet.hub.beep(600, 150)
        except Exception:
            pass
        time.sleep(0.8)
    console.print("[bold green]GO GO GO!!![/bold green]")
    try:
        pet.hub.beep(1000, 400)
    except Exception:
        pass
    
    score = 0
    start_t = time.time()
    last_petted = False
    
    try:
        while time.time() - start_t < 10.0:
            elapsed = time.time() - start_t
            dist = 10
            dist_p = pet.hub.check_connected("Distance Sensor")
            if dist_p:
                dist = pet.hub.sensor_cache[dist_p]["distance"]
            
            petted = (dist < 6)
            if petted and not last_petted:
                score += 1
                try:
                    pet.hub.stop_motor()
                    pet.hub.beep(800, 50)
                    pet.hub.set_led("purple")
                    pet.hub.set_motor(40 if score % 2 == 0 else -40)
                except Exception:
                    pass
            elif not petted:
                try:
                    pet.hub.set_led("green")
                    pet.hub.stop_motor()
                except Exception:
                    pass
                
            last_petted = petted
            console.print(f"Time: {10.0 - elapsed:.1f}s | Pets: [bold cyan]{score}[/bold cyan]     ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    rank = "Bronze" if score < 15 else "Silver" if score < 25 else "Gold" if score < 35 else "Platinum 🏆"
    xp = score * 2
    pet.gain_xp(xp)
    console.print(f"\n\n[bold green]Finished! Total Pets: {score} | Rank: {rank} | XP Gained: {xp} XP![/bold green]")
    console.input("\nPress Enter to return to games menu...")


def run_rhythm_matcher_game(pet):
    console.clear()
    console.print("=== 🥁 Tail-Wag Rhythm Matcher 🥁 ===\n", style="bold cyan")
    console.print("Instructions: Kepler will wag its tail and beep in a rhythm.")
    console.print("You must tap the [bold green]Enter[/bold green] key to match that exact rhythm!")
    console.input("\nPress Enter to start Round 1...")
    
    rounds = 3
    round_num = 1
    xp_won = 0
    
    try:
        while round_num <= rounds:
            import random
            patterns = [
                [0.4, 0.4, 0.8],
                [0.2, 0.2, 0.6, 0.6],
                [0.5, 0.5, 0.2, 0.2, 0.5]
            ]
            pattern = random.choice(patterns)
            
            console.print(f"\n[bold yellow]Round {round_num}: Listen and watch Kepler's rhythm...[/bold yellow]")
            time.sleep(1.0)
            
            timestamps = []
            curr = 0.0
            for delay in pattern:
                time.sleep(delay)
                try:
                    pet.hub.set_led("orange")
                    pet.hub.set_motor(50)
                    time.sleep(0.08)
                    pet.hub.stop_motor()
                    pet.hub.beep(600, 100)
                except Exception:
                    pass
                curr += delay
                timestamps.append(curr)
                time.sleep(0.05)
                try:
                    pet.hub.stop_motor()
                    pet.hub.set_led("green")
                except Exception:
                    pass
                
            console.print("\n[bold cyan]YOUR TURN! Press Enter to match the beats![/bold cyan]")
            user_inputs = []
            start_t = time.time()
            
            for i in range(len(pattern)):
                console.input(f"Tap beat {i+1}/{len(pattern)}... ")
                user_inputs.append(time.time() - start_t)
                try:
                    pet.hub.beep(800, 80)
                except Exception:
                    pass
            
            user_timestamps = []
            if len(user_inputs) > 0:
                offset = user_inputs[0]
                user_timestamps = [t - offset for t in user_inputs]
                ref_offset = timestamps[0]
                ref_timestamps = [t - ref_offset for t in timestamps]
                
                diffs = []
                for u_t, r_t in zip(user_timestamps, ref_timestamps):
                    diffs.append(abs(u_t - r_t))
                avg_diff = sum(diffs) / len(diffs) if len(diffs) > 0 else 1.0
                
                accuracy = max(0, int((1.0 - avg_diff) * 100))
                console.print(f"[bold]Beat Match Accuracy: {accuracy}%[/bold]")
                
                if accuracy >= 70:
                    console.print("[green]Awesome! You matched the rhythm! Kepler is dancing![/green]")
                    try:
                        pet.hub.set_led("green")
                        for _ in range(2):
                            pet.hub.set_motor(70)
                            time.sleep(0.15)
                            pet.hub.set_motor(-70)
                            time.sleep(0.15)
                        pet.hub.stop_motor()
                    except Exception:
                        pass
                    pet.gain_xp(35)
                    xp_won += 35
                else:
                    console.print("[red]Rhythm mismatched. Practice makes perfect![/red]")
                    try:
                        pet.hub.beep(300, 300)
                    except Exception:
                        pass
            
            round_num += 1
            if round_num <= rounds:
                console.input("\nPress Enter to begin the next round...")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    console.print(f"\n[green]Game Over! Total XP gained: {xp_won} XP![/green]")
    console.input("\nPress Enter to return to games menu...")


def run_balance_tail_game(pet):
    console.clear()
    console.print("=== ⚖️ Balance the Tail (Tilt Control) ⚖️ ===\n", style="bold cyan")
    console.print("Instructions: Kepler's tail will drift left or right.")
    console.print("Tilt the Smart Hub (Tilt Sensor) in the [bold yellow]opposite[/bold yellow] direction to balance it!")
    console.print("Keep the tail balanced in the center zone for 15 seconds to win!")
    console.input("\nPress Enter to start balancing...")
    
    score = 0
    start_t = time.time()
    balance_zone = 0.0
    
    try:
        while time.time() - start_t < 15.0:
            elapsed = time.time() - start_t
            
            tilt_p = pet.hub.check_connected("Tilt Sensor")
            tilt_val = pet.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
            
            import random
            balance_zone += random.choice([-0.8, -0.4, 0.4, 0.8])
            
            if tilt_val == "Left":
                balance_zone -= 1.2
            elif tilt_val == "Right":
                balance_zone += 1.2
                
            balance_zone = min(6.0, max(-6.0, balance_zone))
            
            pos = int(balance_zone + 6)
            gauge = ["─"] * 13
            gauge[6] = "|"
            gauge[pos] = "❌"
            gauge_str = "".join(gauge)
            
            status = ""
            if abs(balance_zone) <= 2.5:
                status = "[green]Balanced![/green]"
                score += 1
                try:
                    pet.hub.set_led("green")
                    pet.hub.set_motor(20 if pos % 2 == 0 else -20)
                except Exception:
                    pass
            else:
                status = "[red]Out of Balance! TILT THE OTHER WAY![/red]"
                try:
                    pet.hub.set_led("red")
                    pet.hub.stop_motor()
                    pet.hub.beep(400, 80)
                    pet.hub.set_motor(80 if balance_zone > 0 else -80)
                except Exception:
                    pass
                
            console.print(f"Time: {15.0 - elapsed:.1f}s | Position: [{gauge_str}] | {status}      ", end="\r")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    success = (score >= 45)
    if success:
        console.print("\n\n[bold green]🎉 Great job! The tail stayed perfectly balanced! Kepler is happy![/bold green]")
        pet.gain_xp(40)
    else:
        console.print("\n\n[red]Oops! The tail drifted too far. Try again to balance it![/red]")
        try:
            pet.hub.beep(250, 400)
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_sound_memory_game(pet):
    console.clear()
    console.print("=== 🎵 Sound Pitch Memory Matcher 🎵 ===\n", style="bold cyan")
    console.print("Instructions: Kepler will play a sequence of different musical beeps.")
    console.print("Listen carefully to the pitches (1 = Low, 2 = Medium, 3 = High).")
    console.print("You must select the correct sequence from the choices presented!")
    console.input("\nPress Enter to start Round 1...")
    
    rounds = 3
    round_num = 1
    xp_gained = 0
    pitches = {
        "1": (400, "Low"),
        "2": (600, "Medium"),
        "3": (800, "High")
    }
    
    try:
        while round_num <= rounds:
            import random
            seq = [random.choice(["1", "2", "3"]) for _ in range(round_num + 2)]
            
            console.print(f"\n[bold yellow]Round {round_num}: Listen to the note pitches...[/bold yellow]")
            time.sleep(1.0)
            
            for note in seq:
                freq, label = pitches[note]
                try:
                    pet.hub.set_led("orange")
                    pet.hub.beep(freq, 350)
                except Exception:
                    pass
                time.sleep(0.4)
                try:
                    pet.hub.set_led("green")
                except Exception:
                    pass
                
            correct_ans = "".join(seq)
            choices = [correct_ans]
            while len(choices) < 3:
                alt = "".join([random.choice(["1", "2", "3"]) for _ in range(round_num + 2)])
                if alt not in choices:
                    choices.append(alt)
            random.shuffle(choices)
            
            menu_opts = [f"Sequence: {' - '.join(list(c))}" for c in choices]
            sel = choose_option_interactive(f"Round {round_num}: Choose the correct sequence", menu_opts)
            if sel == -1:
                break
                
            chosen = choices[sel]
            if chosen == correct_ans:
                console.print("[green]Correct! Your ears are sharp![/green]")
                try:
                    pet.hub.beep(1200, 150)
                except Exception:
                    pass
                pet.gain_xp(30)
                xp_gained += 30
            else:
                console.print(f"[red]Wrong! The correct sequence was: {' - '.join(list(correct_ans))}[/red]")
                try:
                    pet.hub.beep(250, 450)
                except Exception:
                    pass
                
            round_num += 1
            if round_num <= rounds:
                console.input("\nPress Enter to begin the next round...")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.set_led("green")
        except Exception:
            pass
        
    console.print(f"\n[green]Game Over! Total XP gained: {xp_gained} XP![/green]")
    console.input("\nPress Enter to return to games menu...")


def run_code_breaker_game(pet):
    console.clear()
    console.print("=== 🔐 Pet Code Breaker (Pattern Lock) 🔐 ===\n", style="bold cyan")
    console.print("Instructions: Kepler has locked his treat box with a secret 3-input code.")
    console.print("The code consists of 3 actions in sequence. Actions are:")
    console.print("  P = Wave Hand (< 6cm) | L = Tilt Hub Left | R = Tilt Hub Right")
    console.print("Kepler will flash [bold yellow]Yellow[/bold yellow] for correct parts, and [bold red]Red[/bold red] if any part is wrong.")
    console.print("Passcode deducing limit: 5 attempts total!")
    console.input("\nPress Enter to begin cracking...")
    
    import random
    passcode = [random.choice(["P", "L", "R"]) for _ in range(3)]
    attempts = 5
    solved = False
    
    try:
        for attempt in range(1, attempts + 1):
            console.print(f"\n[bold yellow]Attempt {attempt} of {attempts}[/bold yellow]")
            
            user_code = []
            for step in range(3):
                console.print(f"Perform Action {step+1}/3 (Hold position until beep)...")
                action = None
                while action is None:
                    dist_p = pet.hub.check_connected("Distance Sensor")
                    tilt_p = pet.hub.check_connected("Tilt Sensor")
                    
                    dist = pet.hub.sensor_cache[dist_p]["distance"] if dist_p else 10
                    tilt = pet.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
                    
                    if dist < 6:
                        action = "P"
                    elif tilt == "Left":
                        action = "L"
                    elif tilt == "Right":
                        action = "R"
                    time.sleep(0.1)
                    
                user_code.append(action)
                console.print(f"Recorded action: [bold cyan]{action}[/bold cyan]")
                try:
                    pet.hub.beep(800, 100)
                except Exception:
                    pass
                time.sleep(0.8)
                
            correct_count = 0
            for u, p in zip(user_code, passcode):
                if u == p:
                    correct_count += 1
            
            if correct_count == 3:
                solved = True
                break
            else:
                console.print(f"[red]Access Denied! Match count: {correct_count}/3 correct.[/red]")
                try:
                    pet.hub.set_led("red")
                    pet.hub.beep(300, 300)
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    pet.hub.set_led("green")
                except Exception:
                    pass
                
        if solved:
            console.print("\n[bold green]🔑 Code Cracked! Access Granted! Treat box unlocked! Kepler wags his tail! 🔑[/bold green]")
            try:
                pet.hub.set_led("green")
                for _ in range(4):
                    pet.hub.stop_motor()
                    pet.hub.beep(1000, 100)
                    pet.hub.set_motor(60)
                    time.sleep(0.1)
                    pet.hub.stop_motor()
                    pet.hub.set_motor(-60)
                    time.sleep(0.1)
                pet.hub.stop_motor()
            except Exception:
                pass
            pet.gain_xp(50)
        else:
            console.print(f"\n[red]Locked out! The correct passcode was: {' - '.join(passcode)}[/red]")
            try:
                pet.hub.beep(200, 600)
            except Exception:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_tail_snatcher_game(pet):
    console.clear()
    console.print("=== ⚡ Reaction Time Test: Tail Snatcher ⚡ ===\n", style="bold cyan")
    console.print("Instructions: Kepler's tail will wag back and forth.")
    console.print("Watch carefully! As soon as the tail stops and flashes [bold purple]Purple[/bold purple],")
    console.print("place your hand rapidly in front of the Distance Sensor (< 6cm)!")
    console.input("\nPress Enter to begin...")
    
    rounds = 3
    round_num = 1
    times = []
    
    try:
        while round_num <= rounds:
            console.print(f"\n[yellow]Round {round_num}: Tail is wagging... Stay ready...[/yellow]")
            time.sleep(1.0)
            
            import random
            wag_duration = random.uniform(2.0, 5.0)
            start_wag = time.time()
            while time.time() - start_wag < wag_duration:
                try:
                    pet.hub.set_motor(40 if int(time.time()*5)%2==0 else -40)
                    pet.hub.set_led("orange")
                except Exception:
                    pass
                time.sleep(0.1)
                
            try:
                pet.hub.stop_motor()
                pet.hub.set_led("purple")
                pet.hub.beep(1200, 100)
            except Exception:
                pass
            signal_t = time.time()
            
            reacted = False
            react_time = 9.99
            while time.time() - signal_t < 4.0:
                dist = 10
                dist_p = pet.hub.check_connected("Distance Sensor")
                if dist_p:
                    dist = pet.hub.sensor_cache[dist_p]["distance"]
                if dist < 6:
                    react_time = time.time() - signal_t
                    reacted = True
                    break
                time.sleep(0.01)
                
            if reacted:
                console.print(f"[green]Caught! Reaction time: {react_time*1000:.0f} ms![/green]")
                times.append(react_time)
                try:
                    pet.hub.beep(1000, 100)
                except Exception:
                    pass
            else:
                console.print("[red]Too slow! Tail got away![/red]")
                try:
                    pet.hub.beep(250, 400)
                except Exception:
                    pass
                
            try:
                pet.hub.set_led("green")
            except Exception:
                pass
            round_num += 1
            if round_num <= rounds:
                console.input("\nPress Enter to begin the next round...")
                
        if len(times) > 0:
            avg_time = sum(times) / len(times)
            console.print(f"\n[bold green]🏆 Game Over! Average Reaction Time: {avg_time*1000:.0f} ms! 🏆[/bold green]")
            xp = int(max(10, (1.0 - avg_time) * 100))
            pet.gain_xp(xp)
            console.print(f"[green]XP Gained: {xp} XP![/green]")
        else:
            console.print("\n[red]No successful catches! Practice and try again![/red]")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_sound_dj_game(pet):
    console.clear()
    console.print("=== 🎧 Sound DJ: Interactive Theremin 🎧 ===\n", style="bold cyan")
    console.print("Instructions: Kepler turns into a musical instrument!")
    console.print("Move your hand closer or further from the Distance Sensor to control tone pitch.")
    console.print("  Closer (< 3cm) = High Pitch note")
    console.print("  Further (> 8cm) = Low Pitch note")
    console.print("Play music by moving your hand! Press any key or Ctrl+C to exit.")
    console.input("\nPress Enter to start playing...")
    
    start_t = time.time()
    try:
        while True:
            if is_key_pressed():
                read_key(0.01)
                break
                
            dist = 10
            dist_p = pet.hub.check_connected("Distance Sensor")
            if dist_p:
                dist = pet.hub.sensor_cache[dist_p]["distance"]
                
            if dist >= 10:
                try:
                    pet.hub.set_led("off")
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                freq = 1300 - (dist * 100)
                freq = min(1300, max(300, freq))
                
                colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]
                color = colors[min(len(colors)-1, int(dist * 0.7))]
                
                try:
                    pet.hub.set_led(color)
                    pet.hub.stop_motor()
                    pet.hub.beep(int(freq), 120)
                except Exception:
                    pass
                time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    duration = time.time() - start_t
    xp = min(50, int(duration * 2))
    pet.gain_xp(xp)
    console.print(f"\n[green]Thanks for DJing! You played for {duration:.1f}s. XP Gained: {xp} XP![/green]")
    console.input("\nPress Enter to return to games menu...")


def run_tilt_maze_game(pet):
    console.clear()
    console.print("=== 🌀 Tilt Maze Navigator 🌀 ===\n", style="bold cyan")
    console.print("Instructions: Steer the ball (●) to the exit target (X) inside the grid.")
    console.print("Tilt the Smart Hub (Tilt Sensor) to move in that direction:")
    console.print("  ◀ Left / Right ▶ / ▲ Forward / Backward ▼")
    console.input("\nPress Enter to enter the maze...")
    
    grid_size = 5
    player_x, player_y = 0, 0
    target_x, target_y = 4, 4
    
    moves = 0
    start_t = time.time()
    
    try:
        while (player_x != target_x or player_y != target_y):
            grid = []
            for y in range(grid_size):
                row = []
                for x in range(grid_size):
                    if x == player_x and y == player_y:
                        row.append("●")
                    elif x == target_x and y == target_y:
                        row.append("❌")
                    else:
                        row.append("·")
                grid.append("  ".join(row))
            
            console.clear()
            console.print("=== 🌀 Tilt Maze Navigator 🌀 ===", style="bold cyan")
            console.print(f"Moves: {moves} | Time Elapsed: {time.time()-start_t:.1f}s\n")
            for r in grid:
                console.print(r, style="bold white")
            console.print("\n[yellow]Tilt sensor active. Hold tilt to slide...[/yellow]")
            
            tilt_p = pet.hub.check_connected("Tilt Sensor")
            tilt_val = pet.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
            
            moved = False
            if tilt_val == "Left" and player_x > 0:
                player_x -= 1
                moved = True
            elif tilt_val == "Right" and player_x < grid_size - 1:
                player_x += 1
                moved = True
            elif tilt_val == "Forward" and player_y > 0:
                player_y -= 1
                moved = True
            elif tilt_val == "Backward" and player_y < grid_size - 1:
                player_y += 1
                moved = True
                
            if moved:
                moves += 1
                try:
                    pet.hub.beep(900, 80)
                    pet.hub.set_led("cyan")
                    pet.hub.set_motor(30)
                    time.sleep(0.12)
                    pet.hub.stop_motor()
                    pet.hub.set_led("green")
                except Exception:
                    pass
                time.sleep(0.4)
            else:
                time.sleep(0.1)
                
        elapsed = time.time() - start_t
        console.clear()
        console.print("[bold green]🏆 Exit Reached! Maze Solved! Kepler celebrates! 🏆[/bold green]")
        try:
            pet.hub.set_led("green")
            for _ in range(3):
                pet.hub.stop_motor()
                pet.hub.beep(880, 80)
                pet.hub.set_motor(60)
                time.sleep(0.08)
                pet.hub.stop_motor()
                pet.hub.beep(1047, 80)
                pet.hub.set_motor(-60)
                time.sleep(0.08)
            pet.hub.stop_motor()
        except Exception:
            pass
        
        xp = max(10, int(100 - elapsed - moves))
        pet.gain_xp(xp)
        console.print(f"[green]Moves: {moves} | Time: {elapsed:.1f}s | XP Gained: {xp} XP![/green]")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_keep_away_game(pet):
    console.clear()
    console.print("=== 🛡️ Keep-Away (Avoidance Challenge) 🛡️ ===\n", style="bold cyan")
    console.print("Instructions: Kepler will sweep its tail left and right.")
    console.print("Keep your hand at a safe range of [bold green]8 cm[/bold green] from the Distance Sensor.")
    console.print("If your hand gets too close (< 5cm) or too far, you lose points!")
    console.print("Avoid Kepler's tail sweeps for 10 seconds!")
    console.input("\nPress Enter to start...")
    
    score = 100
    start_t = time.time()
    
    try:
        while time.time() - start_t < 10.0:
            elapsed = time.time() - start_t
            
            try:
                motor_dir = 50 if int(elapsed * 4) % 2 == 0 else -50
                pet.hub.set_motor(motor_dir)
            except Exception:
                pass
            
            dist = 10
            dist_p = pet.hub.check_connected("Distance Sensor")
            if dist_p:
                dist = pet.hub.sensor_cache[dist_p]["distance"]
                
            status = ""
            if dist >= 8:
                status = "[green]Safe Range[/green]"
                try:
                    pet.hub.set_led("green")
                except Exception:
                    pass
            elif dist <= 5:
                status = "[red]Too Close! BACK AWAY![/red]"
                score -= 3
                try:
                    pet.hub.set_led("red")
                    pet.hub.stop_motor()
                    pet.hub.beep(300, 60)
                except Exception:
                    pass
            else:
                status = "[yellow]Warning: Closing In![/yellow]"
                score -= 1
                try:
                    pet.hub.set_led("yellow")
                except Exception:
                    pass
                
            console.print(f"Time: {10.0 - elapsed:.1f}s | Hand distance: {dist}cm | {status}      ", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    score = max(0, score)
    xp = int(score / 2)
    pet.gain_xp(xp)
    console.print(f"\n\n[bold green]Game Complete! Your Score: {score}/100 | XP Gained: {xp} XP![/bold green]")
    console.input("\nPress Enter to return to games menu...")


def run_simon_says_tilt_game(pet):
    console.clear()
    console.print("=== 📐 Simon Says: Tilt Version 📐 ===\n", style="bold cyan")
    console.print("Instructions: Kepler will prompt you with a sequence of tilts.")
    console.print("Tilt the Smart Hub (Tilt Sensor) in that exact sequence to win!")
    console.print("Directions: L = Left | R = Right | F = Forward | B = Backward")
    console.input("\nPress Enter to start Round 1...")
    
    round_num = 1
    playing = True
    directions = ["L", "R", "F", "B"]
    beep_freqs = {"L": 500, "R": 700, "F": 900, "B": 1100}
    led_colors = {"L": "blue", "R": "orange", "F": "green", "B": "red"}
    
    try:
        while playing:
            import random
            seq = [random.choice(directions) for _ in range(round_num + 1)]
            
            console.print(f"\n[bold yellow]Round {round_num}: Watch the sequence...[/bold yellow]")
            time.sleep(1.0)
            
            for direction in seq:
                console.print(f"Kepler tilts: [bold cyan]{direction}[/bold cyan]")
                try:
                    pet.hub.set_led(led_colors[direction])
                    pet.hub.beep(beep_freqs[direction], 400)
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    pet.hub.set_led("green")
                except Exception:
                    pass
                time.sleep(0.2)
                
            console.print("\n[bold cyan]YOUR TURN! Perform the sequence now (hold tilt until beep)...[/bold cyan]")
            
            user_seq = []
            for step in range(len(seq)):
                console.print(f"Action {step+1}/{len(seq)}:")
                action = None
                while action is None:
                    tilt_p = pet.hub.check_connected("Tilt Sensor")
                    tilt_val = pet.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
                    if tilt_val == "Left": action = "L"
                    elif tilt_val == "Right": action = "R"
                    elif tilt_val == "Forward": action = "F"
                    elif tilt_val == "Backward": action = "B"
                    time.sleep(0.1)
                
                user_seq.append(action)
                console.print(f"Detected: [bold green]{action}[/bold green]")
                try:
                    pet.hub.beep(beep_freqs[action], 200)
                except Exception:
                    pass
                time.sleep(0.8)
                
            if user_seq == seq:
                console.print("[green]Correct! Kepler wags his tail in joy![/green]")
                try:
                    pet.hub.set_led("green")
                    pet.hub.set_motor(50)
                    time.sleep(0.3)
                    pet.hub.stop_motor()
                except Exception:
                    pass
                pet.gain_xp(30)
                round_num += 1
                console.input("\nPress Enter to begin the next round...")
            else:
                console.print(f"[red]Wrong! The sequence was: {' - '.join(seq)}[/red]")
                try:
                    pet.hub.beep(250, 500)
                except Exception:
                    pass
                playing = False
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
        
    console.print(f"\n[green]Game Over! Total rounds completed: {round_num - 1}![/green]")
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


def run_tail_counter_game(pet):
    console.clear()
    console.print("=== 🧮 Tail Counter (Counting Math) 🧮 ===\n", style="bold cyan")
    console.print("Instructions: Kepler will wag his tail a secret number of times (between 1 and 5).")
    console.print("Watch and count the wags carefully!")
    console.print("To answer, wave your hand under the Distance Sensor (< 6cm) [bold green]exactly[/bold green] that many times!")
    console.input("\nPress Enter to begin...")
    
    rounds = 3
    round_num = 1
    total_xp = 0
    
    try:
        while round_num <= rounds:
            import random
            secret_count = random.randint(1, 5)
            console.print(f"\n[bold yellow]Round {round_num}: Watch Kepler count...[/bold yellow]")
            time.sleep(1.0)
            
            for _ in range(secret_count):
                try:
                    pet.hub.set_motor(55)
                    time.sleep(0.2)
                    pet.hub.set_motor(-55)
                    time.sleep(0.2)
                except Exception:
                    pass
            try:
                pet.hub.stop_motor()
            except Exception:
                pass
                
            console.print("\n[bold cyan]YOUR TURN! Wave your hand to match the count (Hold hand close then withdraw for each count)...[/bold cyan]")
            user_count = 0
            start_t = time.time()
            last_close = False
            
            while time.time() - start_t < 8.0:
                dist = 10
                dist_p = pet.hub.check_connected("Distance Sensor")
                if dist_p:
                    dist = pet.hub.sensor_cache[dist_p]["distance"]
                    
                is_close = (dist < 6)
                if is_close and not last_close:
                    user_count += 1
                    console.print(f"Recorded Wave: [bold green]{user_count}[/bold green]     ", end="\r")
                    try:
                        pet.hub.stop_motor()
                        pet.hub.beep(800, 80)
                    except Exception:
                        pass
                last_close = is_close
                time.sleep(0.05)
                
            console.print(f"\nTime's up! You petted: {user_count} wags.")
            if user_count == secret_count:
                console.print("[green]Correct! Kepler is doing a happy waddle![/green]")
                try:
                    pet.hub.set_led("green")
                    for _ in range(3):
                        pet.hub.beep(900, 100)
                        time.sleep(0.1)
                except Exception:
                    pass
                pet.gain_xp(30)
                total_xp += 30
            else:
                console.print(f"[red]Incorrect! Kepler counted {secret_count} wags.[/red]")
                try:
                    pet.hub.beep(200, 450)
                except Exception:
                    pass
            
            round_num += 1
            if round_num <= rounds:
                console.input("\nPress Enter to begin the next round...")
                
        console.print(f"\n[green]Game Complete! Total XP gained: {total_xp} XP![/green]")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
    console.input("\nPress Enter to return to games menu...")


def run_tug_of_war_game(pet):
    console.clear()
    console.print("=== 🪢 Pet Tug-of-War (Tilt Pull) 🪢 ===\n", style="bold cyan")
    console.print("Instructions: Kepler is pulling on the rope! The tail is sweeping with force.")
    console.print("Tilt the Smart Hub (Tilt Sensor) Left and Right repeatedly to pull Kepler back!")
    console.print("You need to match 15 correct pull tilts in 12 seconds to win!")
    console.input("\nPress Enter to start pulling...")
    
    target_pulls = 15
    pulls_done = 0
    start_t = time.time()
    last_tilt = "Neutral"
    
    try:
        while time.time() - start_t < 12.0 and pulls_done < target_pulls:
            elapsed = time.time() - start_t
            
            try:
                pet.hub.set_motor(60 if int(elapsed * 4) % 2 == 0 else -60)
            except Exception:
                pass
                
            tilt_p = pet.hub.check_connected("Tilt Sensor")
            tilt_val = pet.hub.sensor_cache[tilt_p]["tilt"] if tilt_p else "Neutral"
            
            if tilt_val in ["Left", "Right"] and tilt_val != last_tilt:
                pulls_done += 1
                try:
                    pet.hub.stop_motor()
                    pet.hub.beep(1000, 50)
                except Exception:
                    pass
                
            last_tilt = tilt_val
            
            filled = min(10, int((pulls_done / target_pulls) * 10))
            empty = 10 - filled
            gauge = "=" * filled + ">" + " " * empty
            
            console.print(f"Time: {12.0 - elapsed:.1f}s | Progress: [{gauge}] {pulls_done}/{target_pulls} pulls     ", end="\r")
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pet.hub.stop_motor()
            pet.hub.set_led("green")
        except Exception:
            pass
            
    if pulls_done >= target_pulls:
        console.print("\n\n[bold green]🏆 Victory! You won the Tug-of-War! Kepler lies down happily! 🏆[/bold green]")
        pet.gain_xp(45)
    else:
        console.print(f"\n\n[red]Defeat! Kepler pulled you over! Total pulls: {pulls_done}/{target_pulls}.[/red]")
        try:
            pet.hub.beep(250, 450)
        except Exception:
            pass
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
    parser.add_argument("--play-game", type=str, default=None,
                        help="Start directly into a specific minigame and exit when done.")
    parser.add_argument("--pet-name", type=str, default=None,
                        help="Name of the pet to load directly.")
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


    no_tty = not sys.stdin.isatty()

    # Resolve Hub Connection
    hub = None
    hub_type = "Physical (BLE)"
    
    if args.mock or no_tty:
        # If no terminal is open, fall back to simulated mock mode by default
        hub = MockWeDo2Hub(args.hub_name)
        hub_type = "Simulated (Mock)"
    else:
        hub, hub_type = select_hub_flow(args.hub_name)


    # Select or initialize Pet state
    state_dict = select_or_create_pet(args.pet_name)
    is_new = state_dict.pop("is_new", False) if "is_new" in state_dict else False
    pet = DeskPet(hub, state_dict)
    
    if args.play_game:
        if args.play_game in GAMES_LIST:
            game_func_name = GAMES_LIST[args.play_game][1]
            if game_func_name:
                game_func = globals().get(game_func_name)
                if game_func:
                    console.print(f"[cyan]Launching {GAMES_LIST[args.play_game][0]} directly...[/cyan]")
                    try:
                        game_func(pet)
                    except Exception as e:
                        console.print(f"[red]Error playing game: {e}[/red]")
                    if pet.wants_game_id == args.play_game:
                        pet.wants_game_id = None
                        pet.save_state()
            else:
                run_interactive_tutorial(pet)
        sys.exit(0)

    if is_new and not no_tty:
        run_interactive_tutorial(pet)

    if no_tty:
        try:
            while pet.is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            pet.is_running = False
            try:
                hub.stop_motor()
            except Exception:
                pass
            hub.disconnect()
    else:
        # Show initial live dashboard
        try:
            run_live_dashboard(pet, hub_type)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting cleanly...[/yellow]")
        finally:
            pet.is_running = False
            try:
                hub.stop_motor()
            except Exception:
                pass
            try:
                hub.set_led("blue")
                time.sleep(0.2)
            except Exception:
                pass
            hub.disconnect()
            console.print("[green]Goodbye![/green]")

if __name__ == "__main__":
    main()
