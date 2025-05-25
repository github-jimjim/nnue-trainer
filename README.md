# How to Train

Follow these simple steps to train your own NNUE network quickly and efficiently!

---

## 1. Prepare Your Training Set
- Option 1: **Generate the training set yourself**
  - Download the AIO package:
    https://github.com/FireFather/sf-nnue-aio/releases/tag/08-01-2022-AIO
  - Open the AIO set.
  - Compile `process.rs` yourself with Rust:
    ```bash
    rustc -C opt-level=3 process.rs
    ```
  - Then run inside the AIO program:
    ```bash
    gensfen depth 8 loop 75000000 save_every 75000000 output_file_name train.bin
    gensfen depth 8 loop 20000000 save_every 20000000 output_file_name val.bin
    ```
- Option 2: **Download pre-generated training sets** from my website if you don't want to generate them manually.
  ´´´bash
  https://jimmy1205.neocities.org/chess/jomfish
  ´´´

---

## 2. Install Requirements
- Install **Python 3.x**
- Install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

---

## 3. Adjust `main.py`
- Open `main.py` in any text editor.
- At the end of the file, find the lines:
  ```bash
  train_loader = 'train_set_1.bin'  # Change this to the file you want to train with
  ```
- Example: If you are training with `train_set_2.bin`, set it accordingly.

---

## 4. Start Training
Run:
```bash
python main.py
```

---

## 5. Continue Training with a New Set
If you have more training files:
1. Train with the first set.
2. Save the checkpoint.
3. To continue, run:
   ```bash
   python train.py --resume checkpoint_last.ckpt
   ```
4. Update `train_loader` in `main.py` to the next file (e.g., `train_set_2.bin`) and continue training.

---

## 6. Quantize
The trainingscode while generate checkpoints in the folder log. 
1. Search for a .ckpt and move it to the root directory.
2. To quantize run:
   ```bash
   python train.py --quantize
   ```
   
---


## 6. Pretrained Model (Optional)
If you don't want to train from scratch, you can use the already trained model `nnue_pretrained.bin` that I uploaded.

---

## Credits
- Thanks to [DanielUranga/TensorFlowNNUE](https://github.com/DanielUranga/TensorFlowNNUE) for the open-source NNUE model!

---

Stay focused and have fun training! 🚀
