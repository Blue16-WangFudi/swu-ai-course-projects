import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class MotionPredictionNetwork(nn.Module):
    """完整运动预测网络"""

    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.num_layers = num_layers

        # ConvLSTM层
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dim))

        # 预测头
        self.pred_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, 1)  # 预测速度/加速度
        )

    def forward(self, x_sequence):
        # x_sequence: (B, T, C, H, W)
        B, T, C, H, W = x_sequence.shape

        # 初始化隐藏状态
        h_states = [torch.zeros(B, self.cells[i].hidden_dim, H, W).to(x_sequence.device)
                    for i in range(self.num_layers)]
        c_states = [torch.zeros(B, self.cells[i].hidden_dim, H, W).to(x_sequence.device)
                    for i in range(self.num_layers)]

        outputs = []
        for t in range(T):
            x_t = x_sequence[:, t]

            # 多层ConvLSTM
            for layer_idx in range(self.num_layers):
                h_states[layer_idx], c_states[layer_idx] = self.cells[layer_idx](
                    x_t if layer_idx == 0 else h_states[layer_idx - 1],
                    (h_states[layer_idx], c_states[layer_idx])
                )
                x_t = h_states[layer_idx]

            # 预测当前帧运动
            pred = self.pred_head(h_states[-1])
            outputs.append(pred)

        # 返回最后一帧的预测
        return torch.stack(outputs, dim=1)