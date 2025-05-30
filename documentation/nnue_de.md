## Vollständige Implementierungs-Dokumentation: NNUE in Jomfish eigenständig entwickeln

Diese Dokumentation beschreibt detailliert **jede einzelne Funktion**, Klassenmethode und jedes Modul der Jomfish-NNUE-Implementierung. Verwende sie als Referenz, um den Code exakt selbst zu schreiben.

---

# Abschnitt A: Module & Klassenübersicht

1. **bit\_reader.py**: `BitReader` (Pufferung & Bitzugriff)
2. **conversions.py**: `twos`, `orient`, `halfkp_idx`
3. **feature.py**: `get_halfkp_indices`
4. **dataset.py**: `ToTensor`, `NNUEBinData`, `LazyNNUEIterable`
5. **model.py**: `NNUE` LightningModule
6. **exporter.py**: `NNUEWriter`, `main()` Trainings-CLI

---

# Abschnitt B: bit\_reader.py – BitReader

## B.1 Klasse `BitReader`

### `__init__(self, data: bytes | mmap, at: int)`

* **Parameter**:

  * `data`: Bytequelle (Bytearray oder Memory-Mapped-Object).
  * `at`: Start-Byte-Offset.
* **Initialisierung**:

  ```python
  self.bytes = data
  self.seek(at)
  ```

### `seek(self, at: int) -> None`

* Setzt Lesezeiger und Puffer:

  ```python
  self.at = at
  self.bits = 0
  self.position = 0
  self.refill()
  ```

### `refill(self) -> None`

* Füllt Puffer bis ≥32 Bits:

  ```python
  while self.position <= 24:
      byte = self.bytes[self.at]
      self.bits |= byte << self.position
      self.position += 8
      self.at += 1
  ```

### `read_bits(self, n: int) -> int`

* Liest n Bits (LSB-first) und gibt Integer zurück:

  ```python
  mask = (1 << n) - 1
  value = self.bits & mask
  self.bits >>= n
  self.position -= n
  return value
  ```

---

# Abschnitt C: conversions.py – Grundfunktionen

## C.1 `twos(v: int, w: int) -> int`

* **Zweierkomplement-Konversion**:

  ```python
  mask = 1 << (w - 1)
  return (v ^ mask) - mask
  ```
* **Beispiel**: `twos(0xFFFE, 16) = -2`

## C.2 `orient(is_white: bool, sq: int) -> int`

* **Brettspiegelung**:

  ```python
  return ((1 - int(is_white)) * 63) ^ sq
  ```

## C.3 `halfkp_idx(is_white: bool, king_sq: int, sq: int, p: Piece) -> int`

* **Parameter**:

  * `p.piece_type`: 1..6 (ohne König: nur 1..5 verwendet)
  * `p.color`: Boolean
* **Berechnung**:

  ```python
  p_idx = (p.piece_type - 1) * 2 + int(p.color != is_white)
  plane_size = 64
  plane_offset = plane_size * 10 + 1  # 641
  return 1 + orient(is_white, sq) + p_idx * plane_size + king_sq * plane_offset
  ```

---

# Abschnitt D: feature.py – Feature-Erzeugung

## D.1 `get_halfkp_indices(board: chess.Board) -> tuple[Tensor, Tensor]`

1. **Erstelle Tensoren**:

   ```python
   white = torch.zeros(41024, dtype=torch.float32)
   black = torch.zeros(41024, dtype=torch.float32)
   ```
2. **Königsfelder**:

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
4. **Rückgabe**: `(white, black)`

---

# Abschnitt E: dataset.py – Datenklassen

## E.1 Klasse `ToTensor`

### `__call__(self, sample: tuple) -> tuple`

* **Input**: `(board, _, outcome, score)`
* **Ausgabe**:

  * `us`, `them`: FloatTensor(\[0/1]) für Zugfarbe.
  * `widx`, `bidx`: Feature-Vektoren aus D.1.
  * `outcome_t`, `score_t`: FloatTensor.

## E.2 Klasse `NNUEBinData(Dataset)`

### `__init__(self, filename: str, transform=None)`

* **Parameter**:

  * `filename`: Pfad zu `.bin`.
  * `transform`: Funktion, default `ToTensor()`.
* **Berechnung `self.total`**:

  ```python
  size = os.path.getsize(filename)
  self.total = size // 40
  ```

### `_read_raw(self, idx: int) -> tuple`

* Öffnet Dateimapping, instanziiert `BitReader` an `idx*40`.
* Liest sequenziell:

  1. Zugfarbe (1 Bit)
  2. Weiß-König (6 Bit), Schwarz-König (6 Bit)
  3. Felder: für jede der 64 Positionen:

     * Wenn Königsfeld: skip
     * Sonst: LäsBits(1) Präsenz; wenn 1: LäsBits(3) Huffman-Figur, LäsBits(1) Farbe
  4. Score: `twos(br.read_bits(16), 16)`
  5. Ein 16-Bit-Feld überspringen, refill, 16-Bit skip
  6. Outcome: 8 Bit → Map {1:1.0,0:0.5,255:0.0}
* Rückgabe: `(board, None, outcome, score)`

### `__getitem__(self, idx: int)`

* Ruft `_read_raw(idx)` und `transform(...)` auf, return

### `__getstate__(self)`

* Entfernt `file` und `bytes` vor Serialisierung für Multiprocessing.

## E.3 Klasse `LazyNNUEIterable(IterableDataset)`

### `__init__(self, files: list[str], transform=None)`

* Speichert Liste von `.bin`-P faden.

### `__iter__(self)`

* Teilt `files` auf Worker-Shards auf via `get_worker_info()`.
* Für jede Datei: Instanziiere `NNUEBinData`, yield samples, close file.

---

# Abschnitt F: model.py – Lightning-Modul

## F.1 Klasse `NNUE(pl.LightningModule)`

### `__init__(self, lr, lr_step, lr_gamma)`

* **Layer**:

  ```python
  self.quant = QuantStub()
  self.dequant = DeQuantStub()
  self.input = nn.Linear(41024, 256)
  self.act = nn.ReLU()
  self.l1 = nn.Linear(512, 32)
  self.l2 = nn.Linear(32, 32)
  self.out = nn.Linear(32, 1)
  ```
* Speichere Hyperparams: `self.save_hyperparameters()`

### `forward(self, us, them, w, b) -> Tensor`

1. Quantisierung:

   ```python
   uq, wq = self.quant(us), self.quant(w)
   tq, bq = self.quant(them), self.quant(b)
   ```
2. Feature-Transform:

   ```python
   w0 = self.input(wq)  # Bias+Gewicht
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
4. FC-Schichten:

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

* Extrahiere `score`, compute `target = sigmoid(0.0016 * score)`.
* `pred = self(us, them, w, b)`; `loss = F.mse_loss(pred, target)`; logge per Epoche.

### `training_step(self, batch, idx)` & `validation_step(self, batch, idx)`

* Rufen `step(batch, 'train')` bzw. `step(batch, 'val')` auf.

### `configure_optimizers(self)`

* `opt = Adadelta(self.parameters(), lr=self.hparams.lr)`
* `sched = StepLR(opt, step_size=self.hparams.lr_step, gamma=self.hparams.lr_gamma)`
* Rückgabe dict mit optimizer, scheduler, monitor='val'

---

# Abschnitt G: exporter.py & CLI

## G.1 Klasse `NNUEWriter`

### `__init__(self, model: NNUE)`

* `self.buf = bytearray()`; `write_header()`; schreibe Magic und Descr.
* `write_feature_transformer(model.input)`, `write_fc_layer(l1)`, `write_fc_layer(l2)`, `write_fc_layer(out, is_output=True)`

### `write_header(self)`

* `int32(0x4A4F4D46)` ('JOMF'), `int32(0x00010001)` Version, Länge & Textbeschreibung.

### `write_feature_transformer(self, layer: nn.Linear)`

* Bias16: `(layer.bias * 127).round().to(int16)` → bytes
* Gewichte16: `weight.T * 127 → int16` → bytes

### `write_fc_layer(self, layer: nn.Linear, is_output=False)`

* **Skalen**:

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

## G.2 Funktion `main()` – CLI

1. parse argparse-Flags (`sets-dir`, `val-file`, `batch-size`, `lr`, `quantize`, `resume`, ...).
2. Falls `quantize`:

   ```python
   for ckpt in sorted(glob):
       model = NNUE.load_from_checkpoint(ckpt)
       writer = NNUEWriter(model.eval())
       write .jnn file
   return
   ```
3. Sonst:

   * Lese `.bin`-Dateien, erstelle `LazyNNUEIterable`, DataLoader für Train/Val.
   * Checkpoint-Callback, Trainer (GPU, bf16, StepLR).
   * `trainer.fit(model, train_loader, val_loader, ckpt_path=resume)`

---

### Abschnitt I: Vollständige Formelsammlung

| Thema       | Formel                                         |
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


## I.2 Modell-Parameteranzahl

Berechne die Gesamtanzahl der Modellparameter mit folgender Formel:

```text
# Parameter pro Layer:
P_input   = INPUTS * L1 + L1                  # Eingangs-Layer (Gewichte + Bias)
P_fc1     = (2 * L1) * L2 + L2               # Fusion FC1 (Gewichte + Bias)
P_fc2     = L2 * L3 + L3                     # Fusion FC2 (Gewichte + Bias)
P_output  = L3 * 1 + 1                       # Output-Layer (Gewichte + Bias)

# Gesamtparameter:
Total_P   = P_input + P_fc1 + P_fc2 + P_output
```