from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import mss
import mss.tools
import os
import json
import subprocess
import threading
import re
import shutil
import glob
import tempfile
from typing import List

from PIL import Image

app = FastAPI()

# --- KONFIGURASI DATASET ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_DIR = os.path.join(DATASET_DIR, "data")
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")
META_DIR = os.path.join(DATASET_DIR, "meta")

FPS_CONFIG = 10
ACTION_DIM_CONFIG = 4
# Screenshot selalu mengikuti active window saat ini.
TARGET_WINDOW_CLASS = ""
# Proporsi crop untuk ambil panel simulator di tengah Roblox Studio.
SIMULATOR_CROP_LEFT_RATIO = 0.1
SIMULATOR_CROP_TOP_RATIO = 0.25
SIMULATOR_CROP_RIGHT_RATIO = 0.01
SIMULATOR_CROP_BOTTOM_RATIO = 0.2

# --- STATE GLOBAL ---
current_episode_id = 1
current_timestep = 0
current_steps_data = []
state_lock = threading.Lock()

# Buat struktur folder jika belum ada
for folder in [DATA_DIR, VIDEOS_DIR, META_DIR]:
    os.makedirs(folder, exist_ok=True)

# Buat info.json global di meta/info.json
meta_info_path = os.path.join(META_DIR, "info.json")


def ensure_meta_info_file():
    os.makedirs(META_DIR, exist_ok=True)

    if not os.path.exists(meta_info_path):
        with open(meta_info_path, "w") as f:
            json.dump({
                "dataset_name": "roblox_vla_dataset",
                "num_episodes": 0,
                "fps": FPS_CONFIG,
                "action_dim": ACTION_DIM_CONFIG,
            }, f, indent=2)


ensure_meta_info_file()


def detect_next_episode_id():
    episode_files = glob.glob(os.path.join(DATA_DIR, "episode_*.json"))
    if not episode_files:
        return 1

    max_episode = 0
    for episode_file in episode_files:
        name = os.path.basename(episode_file)
        number_str = name.replace("episode_", "").replace(".json", "")
        try:
            max_episode = max(max_episode, int(number_str))
        except ValueError:
            continue

    return max_episode + 1


current_episode_id = detect_next_episode_id()

with open(meta_info_path, "r") as f:
    meta_info = json.load(f)

meta_info["num_episodes"] = current_episode_id - 1
with open(meta_info_path, "w") as f:
    json.dump(meta_info, f, indent=2)

# --- MODEL DATA DARI ROBLOX ---
class StepData(BaseModel):
    instruction: str
    action: List[float]
    is_terminal: bool


def run_hyprctl_json(args: list[str]):
    result = subprocess.run(
        ["hyprctl", "-j", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def window_region(window_info: dict):
    position = window_info.get("at") or []
    size = window_info.get("size") or []

    if len(position) != 2 or len(size) != 2:
        raise ValueError("Window geometry tidak valid")

    left, top = position
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Window size tidak valid")

    return {
        "left": int(left),
        "top": int(top),
        "width": int(width),
        "height": int(height),
    }


def get_window_by_class(window_class: str):
    clients = run_hyprctl_json(["clients"])

    for client in clients:
        client_class = (client.get("class") or "").strip()
        if client_class.lower() == window_class.lower():
            return client

    raise ValueError(f"Window class '{window_class}' tidak ditemukan")


def get_active_window_address():
    window_info = run_hyprctl_json(["activewindow"])
    return window_info.get("address")


def focus_window_by_class(window_class: str):
    selector = f"class:^{re.escape(window_class)}$"
    subprocess.run(["hyprctl", "dispatch", "focuswindow", selector], check=True)


def focus_window_by_address(address: str):
    subprocess.run(["hyprctl", "dispatch", "focuswindow", f"address:{address}"], check=True)


def capture_with_grim(filepath, capture_region):
    geometry = f"{capture_region['left']},{capture_region['top']} {capture_region['width']}x{capture_region['height']}"
    subprocess.run(["grim", "-g", geometry, filepath], check=True)


def crop_simulator_center(source_path: str, target_path: str):
    with Image.open(source_path) as image:
        width, height = image.size

        base_left = int(width * SIMULATOR_CROP_LEFT_RATIO)
        base_top = int(height * SIMULATOR_CROP_TOP_RATIO)
        base_right = int(width * (1 - SIMULATOR_CROP_RIGHT_RATIO))
        base_bottom = int(height * (1 - SIMULATOR_CROP_BOTTOM_RATIO))

        if base_right <= base_left or base_bottom <= base_top:
            raise ValueError("Crop simulator tidak valid")

        crop_width = base_right - base_left
        crop_height = base_bottom - base_top
            # Gunakan width sebagai ukuran square, height sesuaikan ke width
        square_size = crop_width
        
        left = base_left + (crop_width - square_size) // 2
        top = base_top + (crop_height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        cropped = image.crop((left, top, right, bottom))
        cropped.save(target_path)


def capture_active_window(filepath):
    active_window = run_hyprctl_json(["activewindow"])
    capture_with_grim(filepath, window_region(active_window))


def capture_window_class(filepath, window_class):
    window = get_window_by_class(window_class)
    previous_active_address = get_active_window_address()

    try:
        focus_window_by_class(window_class)
        capture_with_grim(filepath, window_region(window))
    finally:
        if previous_active_address:
            try:
                focus_window_by_address(previous_active_address)
            except Exception:
                pass

def append_step(image_path: str, data: StepData, timestep: int):
    return {
        "timestep": timestep,
        "observation": {"image": image_path},
        "action": data.action,
        "is_terminal": data.is_terminal,
    }


def build_episode_payload(episode_id: int, instruction: str, num_steps: int, steps: list[dict]):
    return {
        "episode_id": episode_id,
        "instruction": instruction,
        "metadata": {
            "success": True,
            "num_steps": num_steps,
            "fps": FPS_CONFIG,
        },
        "steps": steps,
    }

def take_screenshot(episode_str, timestep):
    """Menjepret active window saat ini lalu crop ke area simulator di tengah."""
    episode_folder = os.path.join(VIDEOS_DIR, episode_str)
    os.makedirs(episode_folder, exist_ok=True)

    filename = f"frame_{timestep:03d}.png"
    filepath = os.path.join(episode_folder, filename)

    if shutil.which("grim"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            capture_active_window(temp_path)
            crop_simulator_center(temp_path, filepath)
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

        return f"videos/{episode_str}/{filename}"

    raise RuntimeError("grim tidak tersedia, active-window capture tidak bisa dijalankan")


def finalize_episode(episode_str: str, data: StepData, num_steps: int, steps: list[dict]):
    # Guard in case dataset folders are removed while server is running.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    ensure_meta_info_file()

    episode_json = build_episode_payload(current_episode_id, data.instruction, num_steps, steps)
    json_filename = os.path.join(DATA_DIR, f"{episode_str}.json")

    with open(json_filename, "w") as f:
        json.dump(episode_json, f, indent=2)

    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)

    meta_info["num_episodes"] = current_episode_id
    with open(meta_info_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"✅ Tersimpan: {json_filename} dengan {current_timestep + 1} steps.")

@app.post("/record")
async def record_step(data: StepData):
    global current_episode_id, current_timestep, current_steps_data

    with state_lock:
        try:
            episode_id = current_episode_id
            timestep = current_timestep
            episode_str = f"episode_{episode_id:04d}"

            image_path = take_screenshot(episode_str, timestep)
            step_info = append_step(image_path, data, timestep)
            current_steps_data.append(step_info)

            if data.is_terminal:
                finalize_episode(episode_str, data, timestep + 1, current_steps_data)

                current_episode_id += 1
                current_timestep = 0
                current_steps_data = []
            else:
                current_timestep += 1

            return {"status": "ok", "timestep": current_timestep}
        except Exception as exc:
            print(f"❌ record_step gagal: {exc}")
            return {"status": "error", "reason": str(exc)}

if __name__ == "__main__":
    # Menjalankan server di port 5000
    uvicorn.run(app, host="0.0.0.0", port=5000)