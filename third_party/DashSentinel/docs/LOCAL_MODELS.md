# Local model integration

DashSentinel can now fuse landmark features with an optional local ONNX model. This is meant to answer the product-readiness concern that facial landmarks alone are not robust enough for a driver monitoring system.

No model is included in the repository. The project stays runnable without one, but production testing should use a locally stored model trained for driver drowsiness/distraction under real cabin lighting.

## Run with a local ONNX model

```bash
python3 run_dashsentinel.py \
  --show-ui \
  --max-faces 3 \
  --local-model-path ./models/dms_classifier.onnx \
  --local-model-input-size 224
```

To require a successfully loaded model before the app starts active scoring:

```bash
python3 run_dashsentinel.py --require-local-model --local-model-path ./models/dms_classifier.onnx
```

## Expected model outputs

The OpenCV DNN adapter accepts common classifier shapes:

- `1` output: drowsy/unsafe probability or logit
- `2` outputs: `[alert, unsafe]`
- `3+` outputs: `[alert, warning, drowsy, ...]`

The app converts this to:

- `model_drowsy_score`
- `model_confidence`
- `model_label`

Those values are fused with eye/mouth/head-pose landmarks only when the model prediction is confident enough.

## Why local only?

Real-time driver monitoring should not depend on a network round trip. Local inference also avoids sending driver video frames to a server. This adapter uses OpenCV DNN so a small ONNX model can run on the same laptop/edge computer that processes the camera stream.
