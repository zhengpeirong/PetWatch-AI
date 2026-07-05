# PetWatch AI

PetWatch AI is a single-page hackathon demo website for an early-warning health monitoring app for pets. The project was created as an entry for [SEA Hack](https://luma.com/jqwg2puv?tk=6EUDvp), an AI Summit event in Hong Kong featuring startup showcases and a hackathon.

## Authors

- Peirong ZHENG — peirong.zheng@connect.polyu.hk
- Ziqi GONG — gongziqi85@gmail.com

## Project Overview

PetWatch AI helps pet owners notice visible health changes earlier by turning short daily videos of a dog or cat into simple, owner-friendly risk summaries. The concept focuses on visible signals such as gait, posture, breathing, activity level, and repetitive behaviors.

The product is intentionally positioned as an early-warning assistant, not a veterinary diagnosis tool. It does not replace professional veterinary care. Instead, it helps owners decide whether to keep monitoring, record another daily video, or contact a vet when visible risk signals appear more concerning.

## Key Features

- A polished landing page for a 3-minute hackathon pitch
- A local interactive demo with simulated video analysis results
- Pet type and scenario selection for dogs and cats
- Risk levels for gait, breathing, and behavior
- Owner-friendly next-step guidance
- Responsible medical disclaimer language throughout the page
- Fully offline frontend implementation

## Demo Scenarios

The interactive demo includes simulated results for:

- Normal daily check
- Possible limping
- Fast resting breathing
- Low activity and discomfort
- Excessive paw licking

## Technical Implementation

This repository now contains a frontend demo plus a lightweight local backend:

- Single `index.html` file
- HTML, CSS, and vanilla JavaScript only
- `backend.py` FastAPI server for real video upload analysis
- No build tool
- No npm install
- Poe OpenAI-compatible API integration
- `ffmpeg`/`ffprobe` frame extraction

## Running the backend demo

Install `ffmpeg` first:

```bash
brew install ffmpeg
```

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure Poe credentials:

```bash
cp .env.example .env
```

Edit `.env` and set `POE_API_KEY`. `POE_MODEL` defaults to the model shown in `.env.example`; change it to any Poe model/bot available to your account.

Start the local server:

```bash
uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser, upload a short pet video, and the backend will:

1. Accept the video upload.
2. Extract three JPEG frames from the clip.
3. Send the frames plus a PetWatch-specific prompt to Poe.
4. Return structured risk cards and owner-friendly next-step guidance.

Extracted frames are cached under `cache/frames/<cache-id>/frame_1.jpg` through `frame_3.jpg`.
The API response also returns `frameUrls` and local `frameFiles` so you can verify exactly what was sent to Poe.

## Product Vision

Future versions of PetWatch AI could include:

- Pose estimation for gait and posture tracking
- Temporal motion analysis across video frames
- Personalized baseline comparison for each pet
- Multi-pet health profiles
- Weekly health summaries
- Vet-ready export reports
- Partnerships with vets, pet insurers, shelters, pet stores, and tele-vet platforms

## Disclaimer

PetWatch AI is a hackathon demo for informational and educational purposes only. It does not provide veterinary diagnosis or medical advice. Pet owners should contact a licensed veterinarian for medical concerns, urgent symptoms, or persistent changes in their pet's behavior or health.
