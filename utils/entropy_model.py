import abc, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from data_compression._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from data_compression import rans
except:
    pass

def pmf_to_quantized_cdf(pmf, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class Softmax:
    def __init__(self, logits):
        super().__init__()
        self.patch_shape = logits.shape[:-1]
        self.pmf_length = logits.shape[-1]
        self.logits = logits

    def prob(self, indexes):
        pmf = F.softmax(self.logits, dim=-1)
        prob = pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return torch.clamp_min(prob, math.log(1e-9))
        else:
            return prob

    def log_prob(self, indexes):
        log_pmf = F.log_softmax(self.logits, dim=-1)
        log_prob = log_pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return torch.clamp_min(log_prob, math.log(1e-9))
        else:
            return log_prob

    def pmf(self):
        pmf = F.softmax(self.logits, dim=-1)

        if pmf.requires_grad:
            return torch.clamp_min(pmf, math.log(1e-9))
        else:
            return pmf

    def log_pmf(self):
        log_pmf = F.log_softmax(self.logits, dim=-1)

        if log_pmf.requires_grad:
            return torch.clamp_min(log_pmf, math.log(1e-9))
        else:
            return log_pmf


class DiscreteEntropyModelBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 prior=None,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        super().__init__()
        self._prior = prior
        self._tail_mass = float(tail_mass)
        self._range_coder_precision = int(range_coder_precision)
        try:
            self._encoder = rans.RansEncoder()
            self._decoder = rans.RansDecoder()
        except:
            pass
        self.register_buffer("_cdf_shape", torch.IntTensor(2).zero_())


    @property
    def prior(self):
        """Prior distribution, used for deriving range coding tables."""
        if self._prior is None:
            raise RuntimeError(
            "This entropy model doesn't hold a reference to its prior "
            "distribution.")
        return self._prior

    @property
    def tail_mass(self):
        return self._tail_mass

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    def _log_prob_from_prior(self, prior, indexes):
        return prior.log_prob(indexes)

    def _log_pmf_from_prior(self, prior):
        return prior.log_pmf()

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

    def compress(self, indexes, cdf_indexes):
        string = self.encode_with_indexes(
            indexes.flatten().int().tolist(),
            cdf_indexes.flatten().int().tolist(),
            self.cdf,
            self.cdf_length,
            self.cdf_offset,
        )

        return string

    def decompress(self, string, cdf_indexes):
        device = cdf_indexes.device
        shape = cdf_indexes.shape
        cdf_indexes = cdf_indexes.flatten().int().tolist()
        values = self.decode_with_indexes(
            string,
            cdf_indexes,
            self.cdf,
            self.cdf_length,
            self.cdf_offset,
        )

        values = torch.tensor(
            values, device=device, dtype=torch.int64).reshape(shape)

        return values

    def get_ready_for_compression(self):
        self._init_tables()
        self._fix_tables()
        self.cdf = self._cdf.int().tolist()
        self.cdf_length = self._cdf_length.flatten().int().tolist()
        self.cdf_offset = self._cdf_offset.flatten().int().tolist()

    def _fix_tables(self):
        cdf, cdf_offset, cdf_length = self._build_tables(
            self.prior, self.range_coder_precision)

        self._cdf_shape.data = torch.IntTensor(list(cdf.shape)).to(cdf.device)
        self._init_tables()
        self._cdf.data = cdf.int()
        self._cdf_offset.data = cdf_offset.int()
        self._cdf_length.data = cdf_length.int()

    def _init_tables(self):
        shape = self._cdf_shape.tolist()
        device = self._cdf_shape.device
        self.register_buffer("_cdf", torch.IntTensor(*shape).to(device))
        self.register_buffer("_cdf_offset", torch.IntTensor(*shape[:1]).to(device))
        self.register_buffer("_cdf_length", torch.IntTensor(*shape[:1]).to(device))

    def _build_tables(self, prior, precision):
        """Computes integer-valued probability tables used by the range coder.
        These tables must not be re-generated independently on the sending and
        receiving side, since small numerical discrepancies between both sides can
        occur in this process. If the tables differ slightly, this in turn would
        very likely cause catastrophic error propagation during range decoding.
        Args:
          prior: distribution
        Returns:
          CDF table, CDF offsets, CDF lengths.
        """
        pmf = prior.pmf()
        num_pmfs, pmf_length = pmf.shape

        pmf_length = pmf_length * torch.ones(num_pmfs).int()
        cdf_length = pmf_length + 2
        cdf_offset = torch.zeros(num_pmfs)

        max_length = pmf_length.max().int().item()

        cdf = torch.zeros(
            [num_pmfs, max_length + 2], dtype=torch.int32)

        for i, p in enumerate(pmf):
            p = p[:pmf_length[i]]
            overflow = (1. - p.sum(dim=0, keepdim=True)).clamp_min(0)
            p = torch.cat([p, overflow], dim=0)
            c = pmf_to_quantized_cdf(p, precision)
            cdf[i, :c.shape[0]] = c
        return cdf, cdf_offset, cdf_length

class DiscreteUnconditionalEntropyModel(DiscreteEntropyModelBase):

    def __init__(self,
                 prior,
                 range_coder_precision=16):
        super().__init__(
            prior=prior,
            range_coder_precision=range_coder_precision)

    def forward(self, indexes):
        log_probs = self._log_prob_from_prior(self.prior, indexes)
        bits = torch.sum(log_probs) / (-math.log(2))
        return bits

    def log_pmf(self,):
        return self._log_pmf_from_prior(self.prior)

    @staticmethod
    def _build_cdf_indexes(shape):
        dims = len(shape)
        B = shape[0]
        C = shape[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(B, 1, *shape[2:])

    def compress(self, indexes):
        cdf_indexes = self._build_cdf_indexes(indexes.shape)
        return super().compress(indexes, cdf_indexes)

    def decompress(self, string, shape):
        shape = shape + (1,)
        cdf_indexes = self._build_cdf_indexes(shape)
        return super().decompress(string, cdf_indexes)