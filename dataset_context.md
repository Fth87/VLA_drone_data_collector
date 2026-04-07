# Dataset Documentation — VLA (π0.5 Compatible)

## Overview

Dataset ini dirancang untuk training model **Vision-Language-Action (VLA)** seperti π0.5 menggunakan format **trajectory-based (episode)**.

Setiap data merepresentasikan urutan interaksi agen dengan environment dalam bentuk:

```
(image_t, instruction, action_t)
```

Struktur ini mengikuti prinsip **RLDS (Reinforcement Learning Dataset)** yang menyimpan data sebagai sequence waktu. ([TechRxiv][1])

---

## Dataset Structure

```
dataset/
├── meta/
│   └── info.json
├── data/
│   ├── episode_0001.json
│   ├── episode_0002.json
├── videos/
│   ├── episode_0001/
│   │   ├── frame_000.png
│   │   ├── frame_001.png
│   ├── episode_0002/
│       ├── frame_000.png
│       ├── frame_001.png
```

---

## Core Concept

Dataset berbasis **episode (trajectory)**:

```
episode
  ├── step_0
  ├── step_1
  ├── step_2
```

Setiap step berisi:

* observation (image)
* action
* instruction (shared per episode)

---

## File Specification

### 1. meta/info.json

Berisi metadata global dataset.

```json
{
  "dataset_name": "roblox_vla_dataset",
  "num_episodes": 50,
  "fps": 10,
  "action_dim": 4
}
```

Field:

| Field        | Deskripsi      |
| ------------ | -------------- |
| dataset_name | Nama dataset   |
| num_episodes | Jumlah episode |
| fps          | Sampling frame |
| action_dim   | Dimensi action |

---

### 2. data/episode_x.json

Berisi satu trajectory lengkap.

```json
{
  "episode_id": 2,
  "instruction": "follow the red cube",
  "metadata": {
    "success": true,
    "num_steps": 5,
    "fps": 10
  },
  "steps": [
    {
      "timestep": 0,
      "observation": {
        "image": "videos/episode_0002/frame_000.png"
      },
      "action": [0.5, 0.0, 0.0, 0.1],
      "is_terminal": false
    },
    {
      "timestep": 1,
      "observation": {
        "image": "videos/episode_0002/frame_001.png"
      },
      "action": [0.6, 0.1, 0.0, 0.1],
      "is_terminal": false
    },
    {
      "timestep": 2,
      "observation": {
        "image": "videos/episode_0002/frame_002.png"
      },
      "action": [0.7, 0.1, 0.0, 0.0],
      "is_terminal": false
    },
    {
      "timestep": 3,
      "observation": {
        "image": "videos/episode_0002/frame_003.png"
      },
      "action": [0.6, 0.0, 0.0, -0.1],
      "is_terminal": false
    },
    {
      "timestep": 4,
      "observation": {
        "image": "videos/episode_0002/frame_004.png"
      },
      "action": [0.5, -0.1, 0.0, -0.2],
      "is_terminal": true
    }
  ]
}
```

---

## Field Explanation

### episode_id

ID unik untuk setiap trajectory.

---

### instruction

Instruksi bahasa natural untuk seluruh episode.

Contoh:

```
"follow the red cube"
```

Digunakan sebagai conditioning input ke model.

---

### steps (array)

Array berurutan yang merepresentasikan waktu.

```
steps[t] → kondisi pada waktu t
```

Setiap elemen berisi:

#### timestep

Index waktu (harus urut, tidak boleh lompat)

#### observation.image

Path ke frame image

#### action

Kontrol agent.

Contoh (drone):

```
[vx, vy, vz, yaw]
```

#### is_terminal

Menandai step terakhir.

* false → masih lanjut
* true → episode selesai

---

## Data Constraints

Dataset harus memenuhi:

1. Steps berurutan (timestep meningkat)
2. Image sesuai dengan action (sinkron)
3. Satu instruction per episode
4. Frame tidak boleh hilang

---

## Training Interpretation

Saat training, data akan diproses menjadi:

```
input:
  image_t + instruction

target:
  action_t → action_t+k
```

Model π0.5 menggunakan:

* action horizon (multi-step prediction)
* multimodal input (vision + language)

([GitHub][2])

---

## Dataset Characteristics

| Property   | Value                     |
| ---------- | ------------------------- |
| Format     | RLDS-style                |
| Data Type  | Sequence (trajectory)     |
| Modalities | Image + Language + Action |
| Structure  | Episode-based             |

---

## Recommended Configuration

Untuk dataset kecil (prototype):

```
episodes: 20–50
steps per episode: 50–100
fps: 5–10
action_dim: 4
```

---

## Notes

* Dataset tidak harus dari robot asli
* Data synthetic tetap valid selama struktur benar
* Kualitas lebih penting daripada realism

---

## Summary

Dataset ini harus mengikuti pola:

```
Episode → Steps → (Image, Action, Instruction)
```

Bukan dataset klasifikasi atau image statis.

---