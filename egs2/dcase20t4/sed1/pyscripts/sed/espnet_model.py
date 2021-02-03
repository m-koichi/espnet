from functools import reduce
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import torch
from typeguard import check_argument_types

from egs2.dcase20t4.sed1.pyscripts.sed.frontend.abs_frontend import AbsFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from egs2.dcase20t4.sed1.pyscripts.sed.abs_sed import AbsSED
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
import torch.nn.functional as F

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSEDModel(AbsESPnetModel):
    """Sound event detection model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        dataaug,
        normalize: Optional[AbsNormalize],
        prenet,
        encoder,
        classifier,
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    ):
        assert check_argument_types()
        super().__init__()
        self.frontend = frontend
        self.normalize = normalize
        self.dataaug = dataaug
        self.prenet = prenet
        self.encoder = encoder
        self.classifier = classifier
        self.criterion = criterion

    def collect_feats(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        audio = audio[:, : audio_lengths.max()]
        feats, feats_lengths = self._extract_feats(audio, audio_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _extract_feats(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio_lengths.dim() == 1, audio_lengths.shape

        # for data-parallel
        audio = audio[:, : audio_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(audio, audio_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = audio, audio_lengths
        return feats, feats_lengths

    def compute_loss(self, predicts, target, batch_split):
        ptr = self.prenet.pooling_time_ratio
        target = target[:, ptr//2::ptr, :]
        if batch_split is not None:
            
            strong_label = target[batch_split["strong"]]
            # strong_label = F.avg_pool2d(label[:bs], (ptr, 1))
            weak_label_s = target[batch_split["strong"]].max(dim=1)[0]
            # breakpoint()
            weak_label_w = target[batch_split["weak"]].max(dim=1)[0]
            loss_strong = self.criterion(predicts["strong"][batch_split["strong"]], strong_label)
            loss_weak = (self.criterion(predicts["weak"][batch_split["strong"]], weak_label_s) \
                        + self.criterion(predicts["weak"][batch_split["weak"]], weak_label_w)) / 2
            
        else:
            loss_strong = self.criterion(predicts["strong"], target)
            loss_weak = self.criterion(predicts["weak"], target.max(dim=1)[0])

        loss = loss_strong + loss_weak      
        return loss, loss_strong, loss_weak

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengths: torch.Tensor,
        batch_split: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            audio: (Batch, Length, ...)
            audio_lengths: (Batch, )
            label: (Batch, Length, Class)
            label_lengths: (Batch, )
        """
        assert label_lengths.dim() == 1, label_lengths.shape
        # Check that batch_size is unified
        assert (
            audio.shape[0]
            == audio_lengths.shape[0]
            == label.shape[0]
            == label_lengths.shape[0]
        )
        batch_size = audio.shape[0]
        if audio.dim() == 2:
            # on-the-fly style feature extraction in forward propagation
            with autocast(False):
                # 1. Extract feats
                feats, feats_lengths = self._extract_feats(audio, audio_lengths)

                # 2. Data augmentation for spectrogram
                if self.dataaug is not None and self.training:
                    feats, feats_lengths = self.dataaug(feats, feats_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)
        else:
            # input feature case, which used for same input for semi-supervised learning
            feats, feats_lengths = audio, audio_lengths

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        feats, feats_lengths = self.prenet(feats, feats_lengths)
        # -> encoder input: (Batch, Length2, Dim2)
        # feats = feats.transpose(0, 1)
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        encoder_out, encoder_out_lens = self.encoder(feats, None)
        # encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        # encoder_out = encoder_out.transpose(0, 1)
        predicts = self.classifier(encoder_out)

        # encoder_out_lens = masks.squeeze(1).sum(1)
        # assert encoder_out.size(0) == audio.size(0), (
        #     encoder_out.size(),
        #     audio.size(0),
        # )
        # assert encoder_out.size(1) <= encoder_out_lens.max(), (
        #     encoder_out.size(),
        #     encoder_out_lens.max(),
        # )

        loss, loss_strong, loss_weak = self.compute_loss(predicts, label, batch_split)

        stats = dict(
            loss=loss.detach(),
            loss_strong=loss_strong.detach(),
            loss_weak=loss_weak.detach(),
        )
        
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return predicts, loss, stats, weight

    # @torch.no_grad()
    # def predict(
    #     self,
    #     audio: torch.Tensor,
    #     audio_lengths: torch.Tensor,
    #     label: torch.Tensor,
    #     label_lengths: torch.Tensor,
    # ) -> Dict:
    #     """Frontend + Encoder. Note that this method is used by asr_inference.py
    #     Args:
    #         audio: (Batch, Length, ...)
    #         audio_lengths: (Batch, )
    #         label: (Batch, Length, Class)
    #         label_lengths: (Batch, )
    #     """
    #     assert label_lengths.dim() == 1, label_lengths.shape
    #     # Check that batch_size is unified
    #     assert (
    #         audio.shape[0]
    #         == audio_lengths.shape[0]
    #         == label.shape[0]
    #         == label_lengths.shape[0]
    #     )
    #     batch_size = audio.shape[0]


    #     with autocast(False):
    #         # 1. Extract feats
    #         feats, feats_lengths = self._extract_feats(audio, audio_lengths)

    #         # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
    #         if self.normalize is not None:
    #             feats, feats_lengths = self.normalize(feats, feats_lengths)

    #     # 3. Forward encoder
    #     # feats: (Batch, Length, Dim)
    #     # -> encoder_out: (Batch, Length2, Dim2)
    #     feats, feats_lengths = self.prenet(feats, feats_lengths)
    #     # -> encoder input: (Length, Batch, Dim)
    #     # feats = feats.transpose(0, 1)
    #     encoder_out, encoder_out_lens = self.encoder(feats, feats_lengths)
    #     # encoder_out = encoder_out.transpose(0,1)
    #     predicts = self.classifier(encoder_out)
    #     # predicts["strong"] = torch.sigmoid(predicts["strong"]).cpu().data.numpy()
    #     # predicts["weak"] = torch.sigmoid(predicts["weak"]).cpu().data.numpy()

    #     assert encoder_out.size(0) == audio.size(0), (
    #         encoder_out.size(),
    #         audio.size(0),
    #     )
    #     assert encoder_out.size(1) <= encoder_out_lens.max(), (
    #         encoder_out.size(),
    #         encoder_out_lens.max(),
    #     )

    #     return predicts
    