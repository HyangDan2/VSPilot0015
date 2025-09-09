# IR Drowsy Detection MVP

A Windows-only PySide6 GUI MVP for working with **IR camera streams**.  
This tool demonstrates:

- **Face bounding box**
- **Eye landmarks**
- **EAR calculation with drowsy alarm**
- **IR Torch control** (if supported by device)
- **Save current frame as PNG with Space key**

The project is structured with OOP modules for maintainability and extension.

---

## 📂 Project Structure

```

ir-drowsy-mvp/
├─ LICENSE
├─ requirements.txt
├─ commit-guide.md
├─ README.md
└─ src/
├─ app.py                # Entry point
├─ capture/ir\_capture.py # IR capture + Torch control
├─ detect/drowsy.py      # Drowsy detection (EAR, landmarks, bbox)
├─ ui/main\_window\.py     # PySide6 GUI
└─ utils/image.py        # SoftwareBitmap → numpy, numpy → QImage

````

---

## ⚙️ Installation

> Python 3.10–3.12 recommended (due to mediapipe compatibility issues).  
> Only works on **Windows** because of WinRT IR camera APIs.

```bash
git clone <repo-url> ir-drowsy-mvp
cd ir-drowsy-mvp

python -m venv .venv
. .venv/Scripts/activate      # Windows PowerShell
# or source .venv/bin/activate # Linux/macOS (capture won't work, UI only)

pip install --upgrade pip
pip install -r requirements.txt
````

---

## ▶️ Run

```bash
python src/app.py
```

---

## 🖥️ Features

* **IR preview only**
* **Face bounding box + Eye landmarks overlay**
* **EAR (Eye Aspect Ratio)** calculation with adjustable threshold
* **Drowsy ON/OFF checkbox** to toggle detection logic
* **IR Torch control (if supported)**

  * Enable checkbox, Power (0–100) spinbox, Apply button
* **Frame saving with Space key** → saves PNG to `result/ir_YYYYmmdd_HHMMSS.png`

---

## ⌨️ Keyboard Shortcuts

| Key       | Action                    |
| --------- | ------------------------- |
| **Space** | Save current frame as PNG |

---

## 📌 Notes & Limitations

* Most **Windows Hello IR cameras** **do not expose `InfraredTorchControl`** → Torch UI will show *“not supported”*.
* For Torch control to work, the IR device must expose the WinRT API or a vendor SDK.
* Mediapipe FaceMesh works with IR grayscale frames, but accuracy depends on IR image quality and illumination.
* Designed as a **prototype/MVP**: not optimized for production or real-time deployment in safety-critical systems.

---

## 📜 License

[MIT License](license.txt)

---
