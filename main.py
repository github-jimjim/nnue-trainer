import os
import sys
import glob
import struct
import random
import mmap
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

def orient(is_white_pov: bool, sq: int):
    return (63 * (not is_white_pov)) ^ sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def get_halfkp_indices(board: chess.Board):
    def piece_indices(turn):
        indices = torch.zeros(INPUTS)
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
        return indices
    return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))

def get_halfkp_indices_sparse(board: chess.Board):
    def piece_indices(turn):
        indices = torch.empty(INPUTS, layout=torch.sparse_coo)
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
        indices.coalesce()
        return indices
    return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))

PACKED_SFEN_VALUE_BYTES = 40
HUFFMAN_MAP = {0b000: chess.PAWN, 0b001: chess.KNIGHT, 0b010: chess.BISHOP, 0b011: chess.ROOK, 0b100: chess.QUEEN}

def twos(v, w):
    return v - int((v << 1) & 2**w)

class BitReader():
    def __init__(self, bytes, at):
        self.bytes = bytes
        self.seek(at)

    def readBits(self, n):
        r = self.bits & ((1 << n) - 1)
        self.bits >>= n
        self.position -= n
        return r

    def refill(self):
        while self.position <= 24:
            self.bits |= self.bytes[self.at] << self.position
            self.position += 8
            self.at += 1

    def seek(self, at):
        self.at = at
        self.bits = 0
        self.position = 0
        self.refill()

def is_quiet(board, from_, to_):
    for mv in board.legal_moves:
        if mv.from_square == from_ and mv.to_square == to_:
            return not board.is_capture(mv)
    return False

class ToTensor(object):
    def __call__(self, sample):
        bd, _, outcome, score = sample
        us = torch.tensor([bd.turn])
        them = torch.tensor([not bd.turn])
        outcome = torch.tensor([outcome])
        score = torch.tensor([score])
        white, black = get_halfkp_indices(bd)
        return us.float(), them.float(), white.float(), black.float(), outcome.float(), score.float()

class RandomFlip(object):
    def __call__(self, sample):
        bd, move, outcome, score = sample
        mirror = random.choice([False, True])
        if mirror:
            bd = bd.mirror()
        return bd, move, outcome, score

class NNUEBinData(torch.utils.data.Dataset):
    def __init__(self, filename, transform=ToTensor()):
        super(NNUEBinData, self).__init__()
        self.filename = filename
        self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
        self.transform = transform
        self.file = None

    def __len__(self):
        return self.len

    def get_raw(self, idx):
        if self.file is None:
            self.file = open(self.filename, 'r+b')
            self.bytes = mmap.mmap(self.file.fileno(), 0)

        base = PACKED_SFEN_VALUE_BYTES * idx
        br = BitReader(self.bytes, base)

        bd = chess.Board(fen=None)
        bd.turn = not br.readBits(1)
        white_king_sq = br.readBits(6)
        black_king_sq = br.readBits(6)
        bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
        bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

        assert(black_king_sq != white_king_sq)

        for rank_ in range(8)[::-1]:
            br.refill()
            for file_ in range(8):
                i = chess.square(file_, rank_)
                if white_king_sq == i or black_king_sq == i:
                    continue
                if br.readBits(1):
                    assert(bd.piece_at(i) is None)
                    piece_index = br.readBits(3)
                    piece = HUFFMAN_MAP[piece_index]
                    color = br.readBits(1)
                    bd.set_piece_at(i, chess.Piece(piece, not color))
                    br.refill()

        br.seek(base + 32)
        score = twos(br.readBits(16), 16)
        move = br.readBits(16)
        to_ = move & 63
        from_ = (move & (63 << 6)) >> 6

        br.refill()
        ply = br.readBits(16)
        bd.fullmove_number = ply // 2

        move = chess.Move(from_square=chess.SQUARES[from_], to_square=chess.SQUARES[to_])

        game_result = br.readBits(8)
        outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]
        return bd, move, outcome, score

    def __getitem__(self, idx):
        item = self.get_raw(idx)
        return self.transform(item)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state.pop('bytes', None)
        return state

L1 = 256
L2 = 32
L3 = 32

def cp_conversion(x, alpha=0.0016):
    return (x * alpha).sigmoid()

class NNUE(pl.LightningModule):
    def __init__(self):
        super(NNUE, self).__init__()
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
        us = self.quant(us)
        them = self.quant(them)
        w_in = self.quant(w_in)
        b_in = self.quant(b_in)
        w = self.input(w_in)
        b = self.input(b_in)
        l0_ = self.input_add.add(self.input_mul.mul(us, torch.cat([w, b], dim=1)),
                                  self.input_mul.mul(them, torch.cat([b, w], dim=1)))
        l0_ = self.input_act(l0_)
        l1_ = self.l1_act(self.l1(l0_))
        l2_ = self.l2_act(self.l2(l1_))
        x = self.output(l2_)
        x = self.dequant(x)
        return x

    def step_(self, batch, batch_idx, loss_type):
        us, them, white, black, outcome, score = batch
        output = self(us, them, white, black)
        loss = F.mse_loss(output, cp_conversion(score))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, 'val_loss')

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, 'test_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
        return optimizer

class NNUEWriter():
    def __init__(self, model):
        self.buf = bytearray()
        self.write_header()
        self.int32(0x5d69d7b8)
        self.write_feature_transformer(model.input)
        self.int32(0x63337156)
        self.write_fc_layer(model.l1)
        self.write_fc_layer(model.l2)
        self.write_fc_layer(model.output, is_output=True)

    def write_header(self):
        self.int32(0x7AF32F17)
        self.int32(0x3e5aa6ee)
        description = b"Neural Network for chess engine Pynfish"
        self.int32(len(description))
        self.buf.extend(description)

    def write_feature_transformer(self, layer):
        bias = layer.bias.data
        bias = bias.mul(127).round().to(torch.int16)
        self.buf.extend(bias.cpu().flatten().numpy().tobytes())
        weight = layer.weight.data
        weight = weight.mul(127).round().to(torch.int16)
        self.buf.extend(weight.transpose(0, 1).cpu().flatten().numpy().tobytes())

    def write_fc_layer(self, layer, is_output=False):
        kWeightScaleBits = 6
        kActivationScale = 127.0
        if is_output:
            kBiasScale = (1 << kWeightScaleBits) * kActivationScale 
        else:
            kBiasScale = 9600.0  
        kWeightScale = kBiasScale / kActivationScale  
        kMaxWeight = 127.0 / kWeightScale  

        bias = layer.bias.data
        bias = bias.mul(kBiasScale).round().to(torch.int32)
        self.buf.extend(bias.cpu().flatten().numpy().tobytes())
        weight = layer.weight.data
        weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
        self.buf.extend(weight.flatten().cpu().numpy().tobytes())

    def int16(self, v):
        self.buf.extend(struct.pack("<h", v))

    def int32(self, v):
        self.buf.extend(struct.pack("<i", v))

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "quantize":
        ckpt_files = glob.glob("*.ckpt")
        if not ckpt_files:
            print("Keine ckpt-Datei gefunden!")
            return
        for ckpt in ckpt_files:
            print(f"Verarbeite {ckpt} ...")
            nnue_model = NNUE.load_from_checkpoint(ckpt)
            writer = NNUEWriter(nnue_model)
            out_filename = ckpt.rsplit('.', 1)[0] + ".jnn"
            with open(out_filename, 'wb') as f:
                f.write(writer.buf)
            print(f"Konvertiert: {ckpt} -> {out_filename}")
    else:
        nnue = NNUE()
        train_data = DataLoader(NNUEBinData('train.bin'), batch_size=2048, shuffle=True, num_workers=5)
        val_data = DataLoader(NNUEBinData('val.bin'), batch_size=512, num_workers=5)
        tb_logger = pl_loggers.TensorBoardLogger('logs/')
        trainer = pl.Trainer(logger=tb_logger, accelerator="gpu", devices="auto", max_epochs=10, precision="bf16-mixed")
        trainer.fit(nnue, train_data, val_data)

if __name__ == '__main__':
    torch.set_num_threads(8)
    print(torch.cuda.is_available())
    torch.set_float32_matmul_precision('high')
    print("Aktuelles Gerät:", torch.cuda.current_device())
    print("Gerätename:", torch.cuda.get_device_name(torch.cuda.current_device()))
    main()
    