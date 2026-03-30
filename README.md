# MR-YOLO
This project proposes MR-YOLO, a lightweight model for blood cell detection in microscopic blood smear images, achieving improved accuracy in complex scenes while maintaining real-time performance.

## Code Structure

- `MR-YOLO/`: Based on YOLOv8-nano, integrated with MB-D2CM, RMSPF, MSFFM, and CSB-QAL.
- Training and evaluation code will be released upon paper acceptance.

## Usage

1. Clone the repo
2. Install requirements
3. Prepare your dataset (format: YOLO)
4. Train:

```bash
python MR-YOLO/train.py --data your_dataset.yaml --cfg your_model.yaml
