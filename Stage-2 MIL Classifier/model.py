import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import smooth_svm_loss

class Attention(nn.Module):
    def __init__(self, num_classes=2, loss="nll", input_dim=None):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 64
        self.K = 1
        self.num_classes = num_classes
        self.loss = loss

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_instances = x.size(1)
        H = x.view(batch_size, num_instances, self.L)

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = F.softmax(A, dim=1)
        M = torch.bmm(A.transpose(1, 2), H)

        Y_prob = self.classifier(M.view(batch_size, -1))
        Y_hat = Y_prob.argmax(dim=1)

        return Y_prob, Y_hat, A.squeeze(2)

    def calculate_classification_error(self, X, Y):
        Y = Y.long()
        _, Y_hat, _ = self.forward(X)
        error = 1. - torch.eq(Y_hat, Y).float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        if self.loss == "nll":
            Y = F.one_hot(Y, num_classes=self.num_classes).float()
            Y_prob, _, A = self.forward(X)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (Y * torch.log(Y_prob)).sum(dim=1)
            neg_log_likelihood = neg_log_likelihood.mean()
            return neg_log_likelihood, A
        elif self.loss == "svm":
            Y = 2 * Y.float().unsqueeze(1) - 1
            Y_prob, _, A = self.forward(X)

            logits = Y_prob[:, 1].unsqueeze(1)
            smooth_loss = smooth_svm_loss(logits, Y, 0.1)

            uniform_dist = torch.full_like(A, 1.0 / A.size(1))
            kl_div_loss = F.kl_div(A.log(), uniform_dist, reduction='batchmean')

            lambda_kl = 0.1
            total_loss = smooth_loss + lambda_kl * kl_div_loss

            return total_loss, A
        

class VarMIL(nn.Module):
    def __init__(self, num_classes=2, loss="nll", input_dim=None):
        super(VarMIL, self).__init__()
        self.L = input_dim
        self.D = 64
        self.K = 1
        self.num_classes = num_classes
        self.loss = loss

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * 2, self.num_classes),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1)
        )

        self.batch_norm_V = nn.BatchNorm1d(self.D)
        self.batch_norm_U = nn.BatchNorm1d(self.D)

    def forward(self, x):
        batch_size = x.size(0)
        num_instances = x.size(1)
        H = x.view(batch_size * num_instances, self.L)

        A_V = self.attention_V(H)
        A_V = self.batch_norm_V(A_V).view(batch_size, num_instances, self.D)
        
        A_U = self.attention_U(H)
        A_U = self.batch_norm_U(A_U).view(batch_size, num_instances, self.D)

        A = self.attention_weights(A_V * A_U)
        A = F.softmax(A, dim=1)

        M = torch.bmm(A.transpose(1, 2), H.view(batch_size, num_instances, self.L))

        if M.dim() == 2:
            M = M.unsqueeze(1)

        M_expanded = M.expand_as(H.view(batch_size, num_instances, self.L))
        var = torch.bmm(A.transpose(1, 2), (H.view(batch_size, num_instances, self.L) - M_expanded) ** 2)
        M_var = torch.cat((M, var), 1).view(batch_size, -1)

        Y_prob = self.classifier(M_var)
        Y_hat = Y_prob.argmax(dim=1)

        return Y_prob, Y_hat, A.squeeze(2)

    def calculate_classification_error(self, X, Y):
        Y = Y.long()
        _, Y_hat, _ = self.forward(X)
        error = 1. - torch.eq(Y_hat, Y).float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        if self.loss == "nll":
            Y = F.one_hot(Y, num_classes=self.num_classes).float()
            Y_prob, _, A = self.forward(X)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (Y * torch.log(Y_prob)).sum(dim=1)
            neg_log_likelihood = neg_log_likelihood.mean()
            return neg_log_likelihood, A
        elif self.loss == "svm":
            Y = 2 * Y.float().unsqueeze(1) - 1
            Y_prob, _, A = self.forward(X)

            logits = Y_prob[:, 1].unsqueeze(1)
            smooth_loss = smooth_svm_loss(logits, Y, 0.1)

            uniform_dist = torch.full_like(A, 1.0 / A.size(1))
            kl_div_loss = F.kl_div(A.log(), uniform_dist, reduction='batchmean')

            lambda_kl = 0.1
            total_loss = smooth_loss + lambda_kl * kl_div_loss

            return total_loss, A