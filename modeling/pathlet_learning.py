"""
Implements pathlet learning algorithm for trajectory data.
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_A(trajectories, all_edges_dict):
    A = np.zeros((len(all_edges_dict), len(trajectories)))
    for j, p in tqdm(list(enumerate(trajectories))):
        taken_edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
        for edge in taken_edges:
            A[all_edges_dict[edge], j] = 1

    return A


def compute_B(all_edges_dict, candidates_pathlets):
    B = np.zeros((len(all_edges_dict), len(candidates_pathlets)))
    for i, p in tqdm(list(enumerate(candidates_pathlets))):
        taken_edges = [(p[j], p[j + 1]) for j in range(len(p) - 1)]
        for edge in taken_edges:
            try:
                B[all_edges_dict[edge], i] = 1
            except KeyError:
                pass
    return B


class PathletLearning:
    def __init__(self, A, B, candidates_pathlets, kwargs):

        self.A_ = torch.from_numpy(A)
        self.B_ = torch.from_numpy(B)
        self.C_ = torch.from_numpy(np.zeros((B.shape[1], A.shape[1])))
        self.C_.requires_grad = True

        self.candidates_pathlets = candidates_pathlets

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.loss_values_term1 = []
        self.loss_values_term2 = []
        self.loss_values_term3 = []

    def cost_term1(self, C):

        value = torch.norm(self.A_ - self.B_ @ C, 2) ** 2 / 2
        return value

    def cost_term2(self, C):
        value = torch.norm(C, 1)
        return value

    def cost_function(self, C):

        v1 = self.cost_term1(C)
        v1_value = v1.item()
        self.loss_values_term1.append(v1_value)

        v2 = self.lambda_ * self.cost_term2(C)
        v2_value = v2.item()
        self.loss_values_term2.append(v2_value)

        loss_sum = v1 + v2
        self.loss_values.append(v1_value + v2_value)

        return loss_sum

    def build_D(self):

        C = self.C_.detach().cpu().numpy()
        mean_values = np.mean(C, axis=1)

        kept_indices = np.argsort(mean_values)[::-1][: self.dictionary_size]
        D = [self.candidates_pathlets[i] for i in kept_indices]
        self.D = D

    def compute_dictionary(self):

        optimizer = torch.optim.Adam([self.C_], lr=0.1)

        self.loss_values = []
        for i in tqdm(range(self.n_steps)):
            loss = self.cost_function(self.C_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.C_.data.clamp_(0, 1)
            if i > 10 and np.mean(self.loss_values[-10:-5]) <= np.mean(
                self.loss_values[-5:]
            ):
                break

        self.build_D()
