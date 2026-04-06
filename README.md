run project:
uv run fastapi dev main.py

API usage:

- POST `/record` = simpan 1 step baru
- `is_terminal: false` = lanjut ke step berikutnya di episode yang sama
- `is_terminal: true` = simpan episode aktif ke file JSON lalu reset state untuk episode baru

Example body:

```json
{
  "instruction": "jalan ke depan",
  "action": [0.0, 1.0, 0.0, 0.0],
  "is_terminal": false
}
```
