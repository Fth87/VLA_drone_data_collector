## Recorder API for Autonomous Control with Vision Language Action Model for Unmanned Aerial Vehicle in in Flight Mission

Run server:

```bash
uv run fastapi dev main.py
```

Endpoint:

- `POST /record` menambah 1 step ke episode aktif.
- `is_terminal=false` melanjutkan episode.
- `is_terminal=true` menyimpan episode ke `dataset/data/episode_XXXX.json` lalu reset state.

Contoh body:

```json
{
  "instruction": "jalan ke depan",
  "action": [0.0, 1.0, 0.0, 0.0],
  "is_terminal": false
}
```

## Export TFDS RLDS

Generate TFRecord shards dari source `dataset/`:

```bash
uv run python export_rlds_tfds.py --overwrite --output-dir dataset_tfds
```

Smoke test (subset episode):

```bash
uv run python export_rlds_tfds.py --overwrite --output-dir dataset_tfds --max-episodes 3
```

Load hasil export:

```python
from rlds_tfds_builder import RobloxVlaRldsBuilder

builder = RobloxVlaRldsBuilder(data_dir="dataset_tfds")
ds = builder.as_dataset(split="train")
```

## Metadata Notes

- Metadata TFDS (`dataset_info.json`, `features.json`) dihasilkan otomatis dari `rlds_tfds_builder.py`.
- Jangan edit file metadata hasil generate secara manual.
- Jika schema builder berubah, regenerate dengan `--overwrite` agar metadata konsisten.
