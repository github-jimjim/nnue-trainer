import os
import sys
import glob
import struct
import random
import mmap
import argparse

import numpy as np
import chess
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)
INPUTS = NUM_PLANES * NUM_SQ
PACKED_SFEN_VALUE_BYTES = 40
HUFFMAN_MAP = {
    0b000: chess.PAWN,
    0b001: chess.KNIGHT,
    0b010: chess.BISHOP,
    0b011: chess.ROOK,
    0b100: chess.QUEEN
}
L1 = 196
L2 = 48
L3 = 32

def orient(is_white_pov: bool, sq: int) -> int:
    return (63 * (not is_white_pov)) ^ sq


def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES


def get_halfkp_indices(board: chess.Board):
    def piece_indices(turn: bool):
        indices = torch.zeros(INPUTS)
        king_sq = orient(turn, board.king(turn))
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            idx = halfkp_idx(turn, king_sq, sq, p)
            indices[idx] = 1.0
        return indices
    return piece_indices(chess.WHITE), piece_indices(chess.BLACK)


def get_halfkp_indices_sparse(board: chess.Board):
    def piece_indices(turn: bool):
        indices = torch.empty(INPUTS, layout=torch.sparse_coo)
        king_sq = orient(turn, board.king(turn))
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            idx = halfkp_idx(turn, king_sq, sq, p)
            indices[idx] = 1.0
        indices.coalesce()
        return indices
    return piece_indices(chess.WHITE), piece_indices(chess.BLACK)
    
def twos(v: int, w: int) -> int:
    return v - int((v << 1) & 2**w)

class BitReader:
    def __init__(self, bytes_obj, at: int):
        self.bytes = bytes_obj
        self.seek(at)

    def readBits(self, n: int) -> int:
        r = self.bits & ((1 << n) - 1)
        self.bits >>= n
        self.position -= n
        return r

    def refill(self):
        while self.position <= 24:
            self.bits |= self.bytes[self.at] << self.position
            self.position += 8
            self.at += 1

    def seek(self, at: int):
        self.at = at
        self.bits = 0
        self.position = 0
        self.refill()

def is_quiet(board: chess.Board, from_sq: int, to_sq: int) -> bool:
    for mv in board.legal_moves:
        if mv.from_square == from_sq and mv.to_square == to_sq:
            return not board.is_capture(mv)
    return False

class ToTensor:
    def __call__(self, sample):
        bd, _, outcome, score = sample
        us = torch.tensor([bd.turn], dtype=torch.float32)
        them = torch.tensor([not bd.turn], dtype=torch.float32)
        outcome = torch.tensor([outcome], dtype=torch.float32)
        score = torch.tensor([score], dtype=torch.float32)
        white, black = get_halfkp_indices(bd)
        return us, them, white, black, outcome, score

class RandomFlip:
    def __call__(self, sample):
        bd, move, outcome, score = sample
        if random.choice([True, False]):
            bd = bd.mirror()
        return bd, move, outcome, score

class NNUEBinData(torch.utils.data.Dataset):
    def __init__(self, filename: str, transform=ToTensor()):
        super().__init__()
        self.filename = filename
        self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
        self.transform = transform
        self.file = None

    def __len__(self) -> int:
        return self.len

    def get_raw(self, idx: int):
        if self.file is None:
            self.file = open(self.filename, 'r+b')
            self.bytes = mmap.mmap(self.file.fileno(), 0)

        base = PACKED_SFEN_VALUE_BYTES * idx
        br = BitReader(self.bytes, base)

        bd = chess.Board(fen=None)
        bd.turn = not bool(br.readBits(1))
        white_king_sq = br.readBits(6)
        black_king_sq = br.readBits(6)
        bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
        bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

        assert white_king_sq != black_king_sq

        for rank in range(8)[::-1]:
            br.refill()
            for file in range(8):
                sq = chess.square(file, rank)
                if sq in (white_king_sq, black_king_sq):
                    continue
                if br.readBits(1):
                    piece_index = br.readBits(3)
                    color = br.readBits(1)
                    piece = HUFFMAN_MAP[piece_index]
                    bd.set_piece_at(sq, chess.Piece(piece, not bool(color)))
                    br.refill()

        br.seek(base + 32)
        score = twos(br.readBits(16), 16)
        move_raw = br.readBits(16)
        from_sq = (move_raw >> 6) & 0x3F
        to_sq = move_raw & 0x3F
        mv = chess.Move(from_square=from_sq, to_square=to_sq)

        br.refill()
        ply = br.readBits(16)
        bd.fullmove_number = ply // 2

        game_result = br.readBits(8)
        outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]
        return bd, mv, outcome, score

    def __getitem__(self, idx: int):
        raw = self.get_raw(idx)
        return self.transform(raw)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state.pop('bytes', None)
        return state


def cp_conversion(x: torch.Tensor, alpha: float = 0.0016) -> torch.Tensor:
    return (x * alpha).sigmoid()

class NNUE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(INPUTS, L1)
        self.input_act = nn.ReLU()
        self.l1 = nn.Linear(2 * L1, L2)
        self.l1_act = nn.ReLU()
        self.l2 = nn.Linear(L2, L3)
        self.l2_act = nn.ReLU()
        self.output = nn.Linear(L3, 1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.input_mul = FloatFunctional()
        self.input_add = FloatFunctional()

    def forward(self, us, them, w_in, b_in):
        us_q = self.quant(us)
        them_q = self.quant(them)
        w_q = self.quant(w_in)
        b_q = self.quant(b_in)

        w = self.input(w_q)
        b = self.input(b_q)
        x = self.input_add.add(
            self.input_mul.mul(us_q, torch.cat([w, b], dim=1)),
            self.input_mul.mul(them_q, torch.cat([b, w], dim=1))
        )
        x = self.input_act(x)
        x = self.l1_act(self.l1(x))
        x = self.l2_act(self.l2(x))
        x = self.dequant(self.output(x))
        return x

    def step_(self, batch, batch_idx, loss_type):
        us, them, white, black, outcome, score = batch
        pred = self(us, them, white, black)
        loss = F.mse_loss(pred, cp_conversion(score))
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, 'val_loss')

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, 'test_loss')

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=1.0)

class NNUEWriter:
    def __init__(self, model: NNUE):
        self.buf = bytearray()
        self.write_header()
        self.int32(0x46495458)
        self.write_feature_transformer(model.input)
        self.int32(0x4C415945)
        self.write_fc_layer(model.l1)
        self.write_fc_layer(model.l2)
        self.write_fc_layer(model.output, is_output=True)

    def write_header(self):
        self.int32(0x4A4F4D46)
        self.int32(0x00010001)
        desc = b"Neural Network for chess engine Jomfish"
        self.int32(len(desc))
        self.buf.extend(desc)

    def write_feature_transformer(self, layer: nn.Linear):
        bias = (layer.bias.data * 127).round().to(torch.int16)
        self.buf.extend(bias.cpu().numpy().tobytes())
        weight = (layer.weight.data * 127).round().to(torch.int16)
        self.buf.extend(weight.transpose(0,1).cpu().numpy().tobytes())

    def write_fc_layer(self, layer: nn.Linear, is_output: bool = False):
        kWeightScaleBits = 6
        kActivationScale = 127.0
        kBiasScale = ((1 << kWeightScaleBits) * kActivationScale) if is_output else 9600.0
        kWeightScale = kBiasScale / kActivationScale
        kMaxWeight = 127.0 / kWeightScale

        bias = (layer.bias.data * kBiasScale).round().to(torch.int32)
        self.buf.extend(bias.cpu().numpy().tobytes())
        weight = layer.weight.data.clamp(-kMaxWeight, kMaxWeight) * kWeightScale
        weight = weight.round().to(torch.int8)
        self.buf.extend(weight.cpu().numpy().tobytes())

    def int32(self, v: int):
        self.buf.extend(struct.pack("<i", v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NNUE Trainer and JNN Converter")
    parser.add_argument("--quantize", action="store_true",
                        help="Convert all .ckpt in cwd to .jnn format")
    parser.add_argument("--resume", type=str,
                        help="Path to a .ckpt checkpoint to resume training")
    args = parser.parse_args()

    if args.quantize:
        ckpts = glob.glob("*.ckpt")
        if not ckpts:
            print("Keine ckpt-Datei gefunden!")
            sys.exit(0)
        for ckpt in ckpts:
            print(f"Verarbeite {ckpt}...")
            model = NNUE.load_from_checkpoint(ckpt)
            writer = NNUEWriter(model)
            out = ckpt.rsplit('.',1)[0] + ".jnn"
            with open(out, 'wb') as f:
                f.write(writer.buf)
            print(f"Konvertiert: {ckpt} -> {out}")
    else:
        if args.resume:
            print(f"Lade Checkpoint {args.resume} zum Weitermodellieren...")
            nnue = NNUE.load_from_checkpoint(args.resume)
        else:
            print("Starte neues Modell von Scratch...")
            nnue = NNUE()

        train_loader = DataLoader(
            NNUEBinData('train_set_1.bin'),
            batch_size=256,            
            shuffle=True,
            num_workers=8,             
            pin_memory=True,           
            prefetch_factor=4,
            persistent_workers=True            
        )
        val_loader = DataLoader(
            NNUEBinData('val.bin'),
            batch_size=512,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        tb_logger = pl_loggers.TensorBoardLogger('logs/')
        trainer = pl.Trainer(
            logger=tb_logger,
            accelerator="gpu",
            devices="auto",
            max_epochs=2,
            precision="bf16-mixed",
            accumulate_grad_batches=4   
        )
        trainer.fit(nnue, train_loader, val_loader)
