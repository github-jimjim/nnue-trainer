import glob
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from .nnue import NNUE
from .data import NNUEBinData
from .quantization import NNUEWriter

def train_nnue(train_file: str,
               val_file: str,
               epochs: int = 10,
               batch_size: int = 2048,
               device: str = "gpu",
               num_workers: int = 5,
               quantize: bool = False):
    print("Starting training... Stay motivated and give it your best!")
    
    nnue = NNUE()
    accelerator = "cpu" if device == "cpu" or not torch.cuda.is_available() else "gpu"
    devices = 1 if accelerator == "gpu" else None

    train_dataset = NNUEBinData(train_file)
    val_dataset = NNUEBinData(val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // 4, num_workers=num_workers, persistent_workers=True)

    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    trainer = pl.Trainer(logger=tb_logger,
                         accelerator=accelerator,
                         devices=devices,
                         max_epochs=epochs,
                         precision="bf16-mixed")
    
    trainer.fit(nnue, train_loader, val_loader)
    
    if quantize:
        print("Quantization in progress. You're on your way to an efficient model!")
        ckpt_files = glob.glob("**/*.ckpt", recursive=True)
        if not ckpt_files:
            print("No checkpoint file found! Skipping quantization.")
        else:
            ckpt = ckpt_files[-1]
            print(f"Processing checkpoint: {ckpt}")
            nnue = NNUE.load_from_checkpoint(ckpt)
            writer = NNUEWriter(nnue)
            out_filename = os.path.splitext(ckpt)[0] + ".jnn"
            with open(out_filename, "wb") as f:
                f.write(writer.buf)
            print(f"Quantization completed: {ckpt} -> {out_filename}")
    
    print("Training complete. Keep it up and stay determined!")
    input()