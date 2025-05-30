import argparse
from .trainer import train_nnue

def main():
    parser = argparse.ArgumentParser(description="Train the NNUE model for your chess engine prototype.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file (e.g., train.bin)")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation data file (e.g., val.bin)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu", help="Device to use for training")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization after training")
    args = parser.parse_args()
    
    train_nnue(train_file=args.train_file,
               val_file=args.val_file,
               epochs=args.epochs,
               batch_size=args.batch_size,
               device=args.device,
               quantize=args.quantize)

if __name__ == "__main__":
    main()
