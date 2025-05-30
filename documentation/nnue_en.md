## Complete Implementation Documentation: Developing NNUE Independently in Jomfish

This documentation describes in detail **every single function**, class method, and module of the Jomfish NNUE implementation. Use it as a reference to write the code exactly yourself.

---

# Section A: Module & Class Overview

1. **bit\_reader.py**: `BitReader` (Buffering & Bit Access)
2. **conversions.py**: `twos`, `orient`, `halfkp_idx`
3. **feature.py**: `get_halfkp_indices`
4. **dataset.py**: `ToTensor`, `NNUEBinData`, `LazyNNUEIterable`
5. **model.py**: `NNUE` LightningModule
6. **exporter.py**: `NNUEWriter`, `main()` Training-CLI

---

# Section B: bit\_reader.py – BitReader

## B.1 Class `BitReader`

### `__init__(self, data: bytes | mmap, at: int)`

* **Parameters**:

  * `data`: Byte source (byte array or memory-mapped object).
  * `at`: Starting byte offset.
* **Initialization**:

  ```python
  self.bytes = data
  self.seek(at)
  ```

### `seek(self, at: int) -> None`

* Sets read pointer and buffer:

  ```python
  self.at = at
  self.bits = 0
  self.position = 0
  self.refill()
  ```

### `refill(self) -> None`

* Fills buffer to ≥32 bits:

  ```python
  while self.position <= 24:
      byte = self.bytes[self.at]
      self.bits |= byte << self.position
      self.position += 8
      self.at += 1
  ```

### `read_bits(self, n: int) -> int`

* Reads n bits (LSB-first) and returns integer:

  ```python
  mask = (1 << n) - 1
  value = self.bits & mask
  self.bits >>= n
  self.position -= n
  return value
  ```

---

# Section C: conversions.py – Basic Functions

## C.1 `twos(v: int, w: int) -> int`

* **Two's Complement Conversion**:

  ```python
  mask = 1 << (w - 1)
  return (v ^ mask) - mask
  ```
* **Example**: `twos(0xFFFE, 16) = -2`

## C.2 `orient(is_white: bool, sq: int) -> int`

* **Board Reflection**:

  ```python
  return ((1 - int(is_white)) * 63) ^ sq
  ```

## C.3 `halfkp_idx(is_white: bool, king_sq: int, sq: int, p: Piece) -> int`

* **Parameters**:

  * `p.piece_type`: 1..6 (without king: only 1..5 used)
  * `p.color`: Boolean
* **Calculation**:

  ```python
  p_idx = (p.piece_type - 1) * 2 + int(p.color != is_white)
  plane_size = 64
  plane_offset = plane_size * 10 + 1  # 641
  return 1 + orient(is_white, sq) + p_idx * plane_size + king_sq * plane_offset
  ```

---

# Section D: feature.py – Feature Generation

## D.1 `get_halfkp_indices(board: chess.Board) -> tuple[Tensor, Tensor]`

1. **Create Tensors**:

   ```python
   white = torch.zeros(41024, dtype=torch.float32)
   black = torch.zeros(41024, dtype=torch.float32)
   ```
2. **King Squares**:

   ```python
   kw = orient(True, board.king(True))
   kb = orient(False, board.king(False))
   ```
3. **Iteration**:

   ```python
   for sq, piece in board.piece_map().items():
       if piece.piece_type == chess.KING: continue
       idx_w = halfkp_idx(True, kw, sq, piece)
       idx_b = halfkp_idx(False, kb, sq, piece)
       white[idx_w] = 1.0
       black[idx_b] = 1.0
   ```
4. **Return**: `(white, black)`

---

# Section E: dataset.py – Data Classes

## E.1 Class `ToTensor`

### `__call__(self, sample: tuple) -> tuple`

* **Input**: `(board, _, outcome, score)`
* **Output**:

  * `us`, `them`: FloatTensor(\[0/1]) for move color.
  * `widx`, `bidx`: Feature vectors from D.1.
  * `outcome_t`, `score_t`: FloatTensor.

## E.2 Class `NNUEBinData(Dataset)`

### `__init__(self, filename: str, transform=None)`

* **Parameters**:

  * `filename`: Path to `.bin`.
  * `transform`: Function, default `ToTensor()`.
* **Calculation `self.total`**:

  ```python
  size = os.path.getsize(filename)
  self.total = size // 40
  ```

### `_read_raw(self, idx: int) -> tuple`

* Opens file mapping, instantiates `BitReader` at `idx*40`.
* Reads sequentially:

  1. Move color (1 bit)
  2. White king (6 bits), black king (6 bits)
  3. Squares: for each of the 64 positions:

     * If king square: skip
     * Otherwise: ReadBits(1) presence; if 1: ReadBits(3) Huffman figure, ReadBits(1) color
  4. Score: `twos(br.read_bits(16), 16)`
  5. Skip a 16-bit field, refill, 16-bit skip
  6. Outcome: 8 bits → Map {1:1.0,0:0.5,255:0.0}
* Return: `(board, None, outcome, score)`

### `__getitem__(self, idx: int)`

* Calls `_read_raw(idx)` and `transform(...)`, return

### `__getstate__(self)`

* Removes `file` and `bytes` before serialization for multiprocessing.

## E.3 Class `LazyNNUEIterable(IterableDataset)`

### `__init__(self, files: list[str], transform=None)`

* Stores list of `.bin` paths.

### `__iter__(self)`

* Divides `files` into worker shards via `get_worker_info()`.
* For each file: Instantiate `NNUEBinData`, yield samples, close file.

---

# Section F: model.py – Lightning Module

## F.1 Class `NNUE(pl.LightningModule)`

### `__init__(self, lr, lr_step, lr_gamma)`

* **Layers**:

  ```python
  self.quant = QuantStub()
  self.dequant = DeQuantStub()
  self.input = nn.Linear(41024, 256)
  self.act = nn.ReLU()
  self.l1 = nn.Linear(512, 32)
  self.l2 = nn.Linear(32, 32)
  self.out = nn.Linear(32, 1)
  ```
* Store Hyperparams: `self.save_hyperparameters()`

### `forward(self, us, them, w, b) -> Tensor`

1. Quantization:

   ```python
   uq, wq = self.quant(us), self.quant(w)
   tq, bq = self.quant(them), self.quant(b)
   ```
2. Feature Transform:

   ```python
   w0 = self.input(wq)  # Bias+Weight
   b0 = self.input(bq)
   ```
3. Fusion & Act:

   ```python
   cat1 = torch.cat([w0,b0], dim=1)
   cat2 = torch.cat([b0,w0], dim=1)
   x = self.act(
       self.fm.add(
           self.fm.mul(uq, cat1),
           self.fm.mul(tq, cat2)
       )
   )
   ```
4. FC Layers:

   ```python
   x = self.act(self.l1(x))
   x = self.act(self.l2(x))
   ```
5. Output + Dequant:

   ```python
   yq = self.out(x)
   return self.dequant(yq)
   ```

### `step(self, batch, tag) -> Tensor`

* Extract `score`, compute `target = sigmoid(0.0016 * score)`.
* `pred = self(us, them, w, b)`; `loss = F.mse_loss(pred, target)`; log per epoch.

### `training_step(self, batch, idx)` & `validation_step(self, batch, idx)`

* Call `step(batch, 'train')` or `step(batch, 'val')`.

### `configure_optimizers(self)`

* `opt = Adadelta(self.parameters(), lr=self.hparams.lr)`
* `sched = StepLR(opt, step_size=self.hparams.lr_step, gamma=self.hparams.lr_gamma)`
* Return dict with optimizer, scheduler, monitor='val'

---

# Section G: exporter.py & CLI

## G.1 Class `NNUEWriter`

### `__init__(self, model: NNUE)`

* `self.buf = bytearray()`; `write_header()`; write Magic and Descr.
* `write_feature_transformer(model.input)`, `write_fc_layer(l1)`, `write_fc_layer(l2)`, `write_fc_layer(out, is_output=True)`

### `write_header(self)`

* `int32(0x4A4F4D46)` ('JOMF'), `int32(0x00010001)` Version, Length & Text Description.

### `write_feature_transformer(self, layer: nn.Linear)`

* Bias16: `(layer.bias * 127).round().to(int16)` → bytes
* Weights16: `weight.T * 127 → int16` → bytes

### `write_fc_layer(self, layer: nn.Linear, is_output=False)`

* **Scales**:

  ```python
  if is_output:
      kBiasScale = (1<<6)*127
  else:
      kBiasScale = 9600.0
  kWeightScale = kBiasScale / 127.0
  ```
* Bias32: `(bias * kBiasScale).round().to(int32)` → bytes
* Weight8: `clamp(weight, ±max) * kWeightScale → int8` → bytes

### `int32(self, v: int)`

* `self.buf.extend(struct.pack('<i', v))`

## G.2 Function `main()` – CLI

1. parse argparse-Flags (`sets-dir`, `val-file`, `batch-size`, `lr`, `quantize`, `resume`, ...).
2. If `quantize`:

   ```python
   for ckpt in sorted(glob):
       model = NNUE.load_from_checkpoint(ckpt)
       writer = NNUEWriter(model.eval())
       write .jnn file
   return
   ```
3. Else:

   * Read `.bin` files, create `LazyNNUEIterable`, DataLoader for Train/Val.
   * Checkpoint-Callback, Trainer (GPU, bf16, StepLR).
   * `trainer.fit(model, train_loader, val_loader, ckpt_path=resume)`

---

### Section I: Complete Collection of Formulas

| Topic       | Formula                                         |
| ----------- | ---------------------------------------------- |
| twos        | `(v ^ (1<<w-1)) - (1<<w-1)`                    |
| orient      | `((1-is_white)*63) XOR sq`                     |
| p\_idx      | `(piece_type-1)*2 + (piece_color != is_white)` |
| halfkp\_idx | `1 + orient + p_idx*64 + king_sq*641`          |
| sigmoid     | `1/(1+e^{-x})`                                 |
| MSE         | `mean((y - t)^2)`                              |
| Quant W\_in | `w_quant = round(W_float * scale_weight_in)`   |
| Quant b\_in | `b_quant = round(b_float * scale_bias_in)`     |
| fusion      | `relu(u * [w0,b0] + t * [b0,w0])`              |
| FC1         | `x1 = relu(W1 @ x + b1)`                       |
| FC2         | `x2 = relu(W2 @ x1 + b2)`                      |
| Output      | `y = Wout @ x2 + bout`                         |
| DeQuant     | `float = quant / scale`                        |
| Parameter   | `P(L1, L2, L3) = (Inputs + 1) * L1 + (2 * L1 +1) + (L1


## I.2 Model Parameter Count

Calculate the total number of model parameters using the following formula:

```text
# Parameters per Layer:
P_input   = INPUTS * L1 + L1                  # Input Layer (Weights + Bias)
P_fc1     = (2 * L1) * L2 + L2               # Fusion FC1 (Weights + Bias)
P_fc2     = L2 * L3 + L3                     # Fusion FC2 (Weights + Bias)
P_output  = L3 * 1 + 1                       # Output Layer (Weights + Bias)

# Total Parameters:
Total_P   = P_input + P_fc1 + P_fc2 + P_output
```
