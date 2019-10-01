from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module
    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate, after_conv=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.after_conv = after_conv
        if after_conv:
            # self.conv = nn.Conv2d(size, size, 3, 1, 1)
            self.pool = nn.MaxPool2d((2, 1))


    def forward(self, x, mask, attn=None):
        """Compute encoded features
        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        nx = self.norm1(x)
        x, attn_ws = self.self_attn(nx, nx, nx, mask)
        x = x + self.dropout(x)
#         x = x + self.dropout(self.self_attn(nx, nx, nx, mask))
        nx = self.norm2(x)
        nx = x + self.dropout(self.feed_forward(nx))
        if self.after_conv:
            # nx = nn.functional.relu(self.conv(nx))
            nx = self.pool(nx)
            mask = mask[:, ::2, ::2]
#            import ipdb
#            ipdb.set_trace()
        return nx, mask, attn_ws
#       return nx, mask
