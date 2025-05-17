# BVB Pose Lab
**Pose Lab** is a lightweight, offline Python toolkit that scores your beach volleyball (BVB) forms and posture.

### 🧱 The Basics

* Extracts a 33‑joint skeleton from a still image with MediaPipe Pose
* Normalises for camera distance and slight tilt
* Compares your joint‑angle vector to a personalised “gold‑standard” reference
* Flags the top‑three joints that deviate beyond adaptive thresholds

### 🏗️ Current Structure

```
pose_lab/
├── .env               # runtime tweaks (paths, thresholds)
├── requirements.txt   # install deps
├── references/        # stored gold‑standard pose vectors
├── data/raw/          # your uploaded photos
├── data/processed/    # landmark & angle dumps (debugging)
└── src/               # modular pipeline code
```
