# 🐾 LEGO WeDo 2.0 CLI Desk Pet

An interactive command-line "Desk Pet" application for LEGO WeDo 2.0. Built with Python, it features live ASCII animations, telemetry, offline neural network fallback brains, system tray minimization support, automated arcade games, and optional Ollama AI integrations.

---

## 🚀 Installation & Setup

To install and run the Desk Pet application directly from this repository:

### 1. Install System Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install rich bleak mido pystray Pillow --break-system-packages
```

### 2. Launch the Application
Navigate to your repository directory and run the main script:
* **BLE Smart Hub Mode (Default)**: Will scan for 3 seconds for nearby LEGO WeDo 2.0 hubs.
  ```bash
  python3 desk_pet.py
  ```
* **Offline Mock Simulation Mode**: Bypasses Bluetooth scanning and runs Kepler inside a hardware emulator.
  ```bash
  python3 desk_pet.py --mock
  ```

---

## 🕹️ How to Interact and Navigate

* **Arrow Key Selection**: The menus (including hub selection, profile changes, games, and the main dashboard command loop) are 100% interactive. Use the **Up/Down Arrow Keys** to highlight options, and press **Enter** to confirm.
* **Background System Tray Daemon**: If you close the terminal window, the application catches `SIGHUP` (Unix/macOS) or `SIGBREAK` (Windows) and detaches. It continues to run in the background (maintaining the WeDo BLE connection and AI autopilot) and minimizes to:
  * **Linux**: Dock / Notification Tray
  * **Windows**: System Tray (near the clock)
  * **macOS**: Top Menu Bar (Status Bar)
* **Restore/Exit Dashboard**: Right-click the system tray icon to choose:
  * **Show Telemetry Dashboard**: Opens a new terminal window displaying the live telemetry layout.
  * **Exit Pet Application**: Safely disconnects the Smarthub and exits the python daemon.

---

## 🧠 Local Neural Network Fallback Brain

If Ollama is offline or not installed, the Desk Pet features a local, self-contained **3-Layer Feed-Forward Neural Network** (`SmallBrainNN`):
* **Autotraining**: Automatically trains at startup in `< 0.02 seconds` via backpropagation.
* **Sensory Classification**: Maps keyword inputs (feed, pet, poke, sing, sleep, play), current vital stats (energy, hunger, Trainer HP), and live sensor inputs to 7 different emotional behaviors (Happy, Angry, Singing, Sleepy, Dizzy, Eating, Waiting).
* **Profile Customization**: Triggers unique sound chirps, motor responses, and speak logs depending on the active profile.

---

## 🎮 Kepler's Games Arcade

Select `[g] Play Games & Live Tutorial` from the main menu to access:
1. **Interactive Live Tutorial**: Walkthrough verifying proximity petting, tilt dizziness triggers, and hunger screaming rules.
2. **Obstacle Course Game**: Drive your WeDo robot car forward; automatically brakes, sound warnings, and reverses if a block is detected within 5cm of the distance sensor.
3. **Color Simon Says**: Replicate sequence patterns flashed on Kepler's LED lights to earn XP.

---

## 🔨 How to Build Standalone Executables

If you want to package the app into a single, self-contained executable file without needing Python installed on the target machine:

### 1. Compile Locally
Run the built-in cross-platform compilation script:
```bash
python3 build_executables.py
```
This script will detect your current operating system and run PyInstaller with the correct hidden BLE imports (`bleak.backends.bluezdbus` for Linux, `bleak.backends.winrt` for Windows, `bleak.backends.corebluetooth` for macOS). The standalone binary will be written to the `dist/` directory.

### 2. Cloud CI/CD Automated Build
The repository contains a pre-configured GitHub Actions pipeline inside `.github/workflows/build.yml`. When pushing code tags (e.g., `v1.0.3`) to GitHub, cloud runners will compile Linux, Windows (`.exe`), and macOS binaries natively and attach them to your GitHub Releases page automatically.
