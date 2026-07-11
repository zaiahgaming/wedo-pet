# Python Global Module: `robot_midi`

The `robot_midi` library has been successfully installed globally for your user environment. It can be imported in any Python 3 script without needing custom path configuration.

- **Full Name**: `robot_midi` (Source file: `robot_midi.py`)
- **Installed Location**: `/home/zaiah/.local/lib/python3.12/site-packages/robot_midi.py`

---

## How to Use the Global Module

You can now import and utilize the module directly from any Python script on your system:

```python
import robot_midi

# Example: Check version or metadata
print("robot_midi module loaded successfully from:", robot_midi.__file__)
```

### Integration with WeDo 2.0

`robot_midi` is designed to parse MIDI files and generate playbacks of frequencies and durations. To use it with your LEGO WeDo 2.0 Hub, you can pass it an adapter with a `beep(frequency_hz, duration_ms)` method:

```python
import time
from wedo2 import WeDo2Hub
import robot_midi

# Initialize your Smarthub connection
hub = WeDo2Hub()
hub.connect()

# Create a WeDo compatible beep adapter
class WeDoAdapter:
    def __init__(self, hub):
        self.hub = hub
    
    def beep(self, freq_hz, duration_ms):
        try:
            self.hub.sound.beeb(freq_hz, duration_ms)
        except Exception as e:
            print("Beep error:", e)

adapter = WeDoAdapter(hub)

# Example: Run a MIDI melody
# robot_midi will compile and sequence the notes to the hub adapter
print("Playing melody...")
```
