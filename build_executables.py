#!/usr/bin/env python3
"""
build_executables.py

Helper script to build self-contained executables for LEGO WeDo 2.0 Desk Pet.
Python applications must be compiled on their target operating system using PyInstaller.
"""

import sys
import os
import subprocess
import shutil

def main():
    print("=== WeDo 2.0 Desk Pet Multi-Platform Executable Builder ===")
    
    current_platform = sys.platform
    print(f"Current OS Platform Detected: {current_platform}")
    
    # Define platform-specific command configuration
    hidden_imports = {
        "linux": "bleak.backends.bluezdbus",
        "win32": "bleak.backends.winrt",
        "darwin": "bleak.backends.corebluetooth"
    }
    
    binary_names = {
        "linux": "desk_pet_linux",
        "win32": "desk_pet_windows.exe",
        "darwin": "desk_pet_macos"
    }
    
    platform_key = "linux"
    if current_platform.startswith("win"):
        platform_key = "win32"
    elif current_platform.startswith("dar"):
        platform_key = "darwin"
        
    hidden_imp = hidden_imports.get(platform_key, "bleak.backends.bluezdbus")
    bin_name = binary_names.get(platform_key, "desk_pet_binary")
    
    # Calculate search paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(workspace_dir)
    
    build_cmd = [
        "pyinstaller",
        "--onefile",
        f"--paths={parent_dir}",
        f"--hidden-import={hidden_imp}",
        f"--name={bin_name}",
        "desk_pet.py"
    ]
    
    print("\n--- Build Instructions for All OS ---")
    print("1. Linux Compilation:")
    print("   pyinstaller --onefile --paths=../ --hidden-import=bleak.backends.bluezdbus --name=desk_pet_linux desk_pet.py\n")
    print("2. Windows Compilation:")
    print("   pyinstaller --onefile --paths=../ --hidden-import=bleak.backends.winrt --name=desk_pet_windows desk_pet.py\n")
    print("3. macOS Compilation:")
    print("   pyinstaller --onefile --paths=../ --hidden-import=bleak.backends.corebluetooth --name=desk_pet_macos desk_pet.py\n")
    
    print("Would you like to build for the current OS now? (y/n)")
    try:
        ans = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        ans = "n"
        
    if ans == "y":
        print(f"\nRunning command: {' '.join(build_cmd)}")
        # Check if PyInstaller is available
        pyinstaller_path = shutil.which("pyinstaller")
        if not pyinstaller_path:
            # Fallback to local user bin
            local_user_path = os.path.expanduser("~/.local/bin/pyinstaller")
            if os.path.exists(local_user_path):
                pyinstaller_path = local_user_path
            else:
                print("[Error] PyInstaller is not installed or not in PATH.")
                print("Please run: pip install pyinstaller")
                return
                
        build_cmd[0] = pyinstaller_path
        try:
            subprocess.run(build_cmd, check=True)
            print("\n[Success] Binary successfully created in 'dist/' directory!")
        except Exception as e:
            print(f"\n[Error] Build failed: {e}")
            
if __name__ == "__main__":
    main()
