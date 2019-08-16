from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_text(self, tag, text, step=None):
        self.writer.add_text(tag, text, step)
