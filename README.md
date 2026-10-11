# SignAR — Filipino Sign Language Decoder

A static, browser-based FSL decoder project. The current foundation tracks hands, pose/body, and selected face landmarks in the browser, normalizes those coordinates, and defines a strict contract for a 20-class temporal TensorFlow.js model.

## Current state

- Camera-first public Decoder Mode.
- Rear camera first, with fallback when it is unavailable; front/rear switching is supported.
- Camera preview stays at the camera's available quality, while frames sent to MediaPipe are downscaled to a maximum dimension of 960 px.
- MediaPipe Holistic draws body, hands, and key facial landmarks.
- Landmark normalization uses a torso anchor/scale and wrist-relative hand geometry.
- Live inference buffers a complete variable-duration sign segment and uses the same linear 30-frame resampling algorithm as the Python trainer.
- TensorFlow.js model loading is supported, but the public repository intentionally has **no trained model yet**.
- No sign labels are fabricated while the trained model is absent.
- The public decoder uses Filipino-only display and prefers a browser Filipino voice, with a local browser speech fallback.
- Raw training recordings are not stored or uploaded by the public decoder.

## Important model status

**This is a foundation, not yet a validated FSL decoder.** A trained temporal model must be produced from verified FSL sign recordings before the camera can reliably decode words. The 20-word list is an initial project vocabulary, not a claim that its signs have been validated. Have the actual forms reviewed by FSL signers, a teacher, or an interpreter before collecting labels.

## Model contract

The public decoder expects:
- model/vocabulary.json — class order and model contract.
- model/model.json and TensorFlow.js weight shards — only after training.
- Input tensor shape [batch, 30, 456].
- Output tensor shape [batch, 20], with class order matching the manifest.

Feature layout:
- 33 pose landmarks × (x, y, z, visibility) = 132 values.
- Left and right hand landmarks, each with 21 body-relative and wrist/palm-relative points = 252 values combined.
- 24 selected face points × (x, y, z) = 72 values.
- Total = 456 values per sampled frame.

The first model is intended for right-handed signers, front-facing sign execution, and one primary signer. Evaluation should split by signer so test participants do not appear in training.

## GitHub Pages

The app can be deployed as a static site from the repository root. Camera permissions require HTTPS (GitHub Pages supplies it). MediaPipe and TensorFlow.js libraries/model assets need an internet connection in this prototype.

## Private training data

Store raw videos in a local/private dataset folder or private backup; do not commit raw videos to the public repository. Export only validated model artifacts and necessary public metadata after review and testing.