import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

EPS = 1e-8

class SiamURE(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, raw_dim=768 * 4, hidden_dim=512, pred_dim=256):
        super(SiamURE, self).__init__()
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder
        self.alpha = 1.0
        # build a 3-layer projector

        prev_dim = 2 * 768
        dim = 768 * 2
        pred_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.LayerNorm(prev_dim),
            nn.LeakyReLU(),
            nn.Linear(prev_dim, prev_dim),
            nn.LayerNorm(prev_dim),
            nn.LeakyReLU(),
            nn.Linear(prev_dim, dim),
            nn.LayerNorm(dim)
        )

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.LayerNorm(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2=None, pat='train'):
        # compute features for one view
        if pat == 'train':
            if x2 is None:
                z1 = self.encoder(sentence_data=x1)  # NxC
                z1_proj = self.fc(z1)

                z2 = self.encoder(sentence_data=x1)  # NxC
                z2_proj = self.fc(z2)

                p1 = self.predictor(z1_proj)  # NxC
                p2 = self.predictor(z2_proj)  # NxC
                return p1, p2, z1_proj.detach(), z2_proj.detach()
            else:
                z1 = self.encoder(sentence_data=x1)  # NxC
                z1_proj = self.fc(z1)

                z2 = self.encoder(sentence_data=x1)  # NxC
                z2_proj = self.fc(z2)

                p1 = self.predictor(z1_proj)  # NxC
                p2 = self.predictor(z2_proj)  # NxC
                z1_neigh = self.encoder(sentence_data=x2)
                return p1, p2, z1_proj.detach(), z2_proj.detach(), z1, z1_neigh

        elif pat == 'test':
            z1 = self.encoder(sentence_data=x1)
            return z1
        else:
            raise NotImplementedError('not implemented!')

    def get_cluster_prob(self, embeddings, cluster_centers):
        cluster_centers = Parameter(cluster_centers)
        #
        norm_squared = torch.sum((embeddings.unsqueeze(1) - cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()
        return t_dist


def entropy(x, input_as_probabilities):
    if input_as_probabilities:
        # torch.clamp ：设置数据域
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0

    def forward(self, anchors, neighbors):
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss
