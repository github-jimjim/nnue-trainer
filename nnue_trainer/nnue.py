import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
import chess

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

def cp_conversion(x, alpha=0.0016):
    return (x * alpha).sigmoid()

L1 = 256
L2 = 32
L3 = 32

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
        l0_ = self.input_add.add(
            self.input_mul.mul(us, torch.cat([w, b], dim=1)),
            self.input_mul.mul(them, torch.cat([b, w], dim=1))
        )
        l0_ = self.input_act(l0_)
        l1_ = self.l1_act(self.l1(l0_))
        l2_ = self.l2_act(self.l2(l1_))
        x = self.output(l2_)
        x = self.dequant(x)
        return x

    def step_(self, batch, batch_idx):
        us, them, white, black, outcome, score = batch
        output = self(us, them, white, black)
        loss = F.mse_loss(output, cp_conversion(score))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step_(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step_(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.step_(batch, batch_idx)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
        return optimizer
