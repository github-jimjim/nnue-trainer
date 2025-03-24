# NNUE Trainer

## Installation

To install the library, use the following command:

```bash
pip install nnue-trainer
```

Additionally, the `nnue-parser` library is required for parsing training data:

```bash
pip install nnue-parser
```


## Usage

A simple example for training an NNUE model:

```python
from nnue_trainer import train_nnue

if __name__ == "__main__":
    train_nnue(
        train_file="train.bin",
        val_file="val.bin",
        epochs=10,
        batch_size=4096,
        device="gpu",
        num_workers=5,
        quantize=True
    )
```

### Parameter Description:

| Parameter      | Description |
|---------------|-------------|
| `train_file`  | Path to the training file in `.bin` format |
| `val_file`    | Path to the validation file in `.bin` format |
| `epochs`      | Number of training iterations |
| `batch_size`  | Batch size for processing |
| `device`      | Choose `"cpu"` or `"gpu"` to specify the device |
| `num_workers` | Number of parallel processes for data loading |
| `quantize`    | If `True`, the model will be quantized |

## Creating `.bin` Training Files

The `.bin` training files can be generated using the following tool:  
[FireFather/sf-nnue-aio](https://github.com/FireFather/sf-nnue-aio/releases/tag/08-01-2022-AIO)

Use the following command:

```bash
gensfen depth 8 loop 100000
```

This command generates SFEN data with a search depth of 8 over 100,000 iterations.

## Parsing the `.jnn` File

After training, the final `.jnn` model file can be parsed using the `nnue-parser` library:

```python
from nnue_parser import parse_nnue

parsed_data = parse_nnue("model.jnn")
```

This allows you to inspect and analyze the trained NNUE model.

## Troubleshooting / Common Errors

### `TypeError: devices selected with CPUAccelerator should be an int > 0.`
**Solution:** Ensure that `devices` in `train_nnue()` is not `None` or `0`. If training on CPU, explicitly set:

```python
train_nnue(..., device="cpu", num_workers=1)
```

### `RuntimeError: CUDA out of memory`
**Solution:** If your GPU memory is insufficient, reduce the `batch_size`:

```python
train_nnue(..., batch_size=1024)
```

## Acknowledgments

A big thank you to the repository [TensorFlowNNUE](https://github.com/DanielUranga/TensorFlowNNUE) for the model.

## License

This project is released under an open-source license. See the `LICENSE` file for more details.

## Release Information

This is the **first and final release** of NNUE Trainer.

