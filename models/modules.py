import torch
from torch import nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, pred, targets):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        # """
        # print('outputs shape:', pred.shape)
        # print('targets shape:', targets.shape)
        
        loss = self.loss(pred, targets)
        accuracy = self._calculate_accuracy(pred, targets)
        return loss, accuracy

    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs, targets):
        correct = self._get_correct(outputs)
        return 100. * (correct == targets).sum().float() / targets.size(0) # True/True


class BinaryClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __name__(self):
        return "BinaryClassificationLoss"
    
    def forward(self, pred, targets):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        pred = pred.float()
        if pred.size().numel() == targets.size().numel():
            targets = targets.float().view(pred.shape)
        else:
            targets = F.one_hot(targets.long(), num_classes=2).float()
        return super().forward(pred, targets)

    def _get_correct(self, outputs):
        return outputs > 0.5


class MixDecoupleClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        if reduction is None:
            self.loss = nn.NLLLoss(weight=weight)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, weight=weight)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)

    def __name__(self):
        return "MixDecoupleClassificationLoss"

class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.CrossEntropyLoss(reduction=reduction,weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(weight=weight)
            
    def __name__(self):
        return "MulticlassClassificationLoss"

    def forward(self, pred, targets):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        targets = targets.squeeze().long()
        return super().forward(pred, targets)
    
    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        """

        :param targets:
        :param outputs:
        :return: a loss value
        """
        raise NotImplementedError()


class CovarianceResidualError(RegressionLoss):  # For Cascade "Correlation"
    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        _, _, graph_emb, errors = outputs

        errors_minus_mean = errors - torch.mean(errors, dim=0)
        activations_minus_mean = graph_emb - torch.mean(graph_emb, dim=0)

        # todo check against commented code
        cov_per_pattern = torch.zeros(errors.shape)

        cov_error = 0.
        for o in range(errors.shape[1]):  # for each output unit
            for i in range(errors.shape[0]):  # for each pattern
                cov_per_pattern[i, o] = errors_minus_mean[i, o]*activations_minus_mean[i, 0]

            cov_error = cov_error + torch.abs(torch.sum(cov_per_pattern[:, o]))

        #print(torch.mean(cov_per_pattern, dim=0), torch.mean(errors_minus_mean), torch.mean(graph_emb))

        '''
        activations_minus_mean = torch.sum(activations_minus_mean, dim=1)
        activations_minus_mean = torch.unsqueeze(activations_minus_mean, dim=1)

        activations_minus_mean = torch.t(activations_minus_mean)

        cov_per_pattern = torch.mm(activations_minus_mean, errors_minus_mean)

        cov_abs = torch.abs(cov_per_pattern)

        # sum over output "units"
        cov_error = torch.sum(cov_abs)
        '''

        # Minus --> maximization problem!
        return - cov_error


class NN4GMulticlassClassificationLoss(MulticlassClassificationLoss):

    def mse(self, ts, ys, return_sum):

        targets_oh = torch.zeros(ys.shape)
        ts = ts.unsqueeze(1)
        targets_oh.scatter_(1, ts, value=1.)  # src must not be specified
        ts = targets_oh

        if return_sum == True:
            return torch.sum(0.5 * (ts - ys) ** 2) / len(ts)
        else:
            return 0.5 * (ts - ys) ** 2 / len(ts)

    def forward(self, targets, *outputs):

        preds, _, _, _ = outputs

        # Try MSE
        loss = self.mse(targets, preds, return_sum=True)

        #loss = self.loss(preds, targets)

        accuracy = self._calculate_accuracy(preds, targets)
        return loss, accuracy


class DiffPoolMulticlassClassificationLoss(MulticlassClassificationLoss):
    """
    DiffPool - No Link Prediction Loss
    """

    def forward(self, targets, *outputs):
        preds, lp_loss, ent_loss = outputs

        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        loss = self.loss(preds, targets)
        accuracy = self._calculate_accuracy(preds, targets)
        return loss + lp_loss + ent_loss, accuracy
