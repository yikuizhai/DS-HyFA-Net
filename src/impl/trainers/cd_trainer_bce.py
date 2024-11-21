import torch

from .cd_trainer import CDTrainer

class CDTrainer_BCE(CDTrainer):

    def _prepare_data(self, t1, t2, tar):
        return super()._prepare_data(t1, t2, tar.float())

    def _process_model_out(self, out):
        out = out.squeeze(1)
        return out

    def _pred_to_prob(self, pred):
        return torch.sigmoid(pred)