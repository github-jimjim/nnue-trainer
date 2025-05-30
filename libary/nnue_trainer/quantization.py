import struct
import torch

class NNUEWriter:
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
        description = b"Neural Network by Jimmy Luong"
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
