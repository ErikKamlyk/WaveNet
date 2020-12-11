import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ConvGateBlock(nn.Module):
    def __init__(self, in_filters, gate_filters, dilation, n_mels):
        super(ConvGateBlock, self).__init__()
        self.dilation = dilation
        self.conv_dilated = nn.Conv1d(in_filters, gate_filters, 2, dilation=dilation)
        self.mels_conv1 = nn.Conv1d(n_mels, gate_filters, 1)
        self.mels_conv2 = nn.Conv1d(n_mels, gate_filters, 1)
        self.gate1 = nn.Conv1d(gate_filters, gate_filters, 1)
        self.gate2 = nn.Conv1d(gate_filters, gate_filters, 1)
        self.conv_final = nn.Conv1d(gate_filters, in_filters, 1)
        self.memory = None

    def forward(self, x, mels, fast_generation=False):
        if fast_generation:
            if self.memory == None:
                self.memory = F.pad(x, (self.dilation, 0))
            else:
                self.memory = torch.cat([self.memory, x], dim=2)
            x1 = self.conv_dilated(self.memory)
            gate1 = torch.tanh(self.gate1(x1) + self.mels_conv1(mels))
            gate2 = torch.sigmoid(self.gate2(x1) + self.mels_conv2(mels))
            out = self.conv_final(gate1*gate2)
            self.memory = self.memory[:, :, x.shape[2]:]
            return x + out, out
        x1 = self.conv_dilated(F.pad(x, (self.dilation, 0)))
        gate1 = torch.tanh(self.gate1(x1) + self.mels_conv1(mels))
        gate2 = torch.sigmoid(self.gate2(x1) + self.mels_conv2(mels))
        out = self.conv_final(gate1*gate2)
        return x + out, out

    def clear_memory(self):
        self.memory = None

class WaveNet(nn.Module):
    def __init__(self, n_mels, n_blocks):
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        filters_in_block = 256
        gate_filters = 256
        filters_out = 256

        self.conv1 = nn.Conv1d(1, filters_in_block, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 10)
            block = ConvGateBlock(filters_in_block, gate_filters, dilation, n_mels)
            self.blocks.append(block)

        self.conv2 = nn.Conv1d(filters_in_block, filters_in_block, 1)

        self.conv3 = nn.Conv1d(filters_in_block, filters_out, 1)
        
    def forward(self, x, mels, fast_generation=False):
        x = self.conv1(x)
        residuals = 0
        for i in range(self.n_blocks):
            x, h = self.blocks[i](x, mels, fast_generation)
            residuals += h
        x = F.relu(self.conv2(F.relu(residuals)))
        x = self.conv3(x)
        return x

    def clear_memory(self):
        for i in range(self.n_blocks):
            self.blocks[i].clear_memory()
