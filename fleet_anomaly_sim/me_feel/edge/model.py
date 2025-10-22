# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiExitResNet(nn.Module):
    """
    A simple multi-exit neural network.
    Internal classifiers (exits) at intermediate layers for early outputs:contentReference[oaicite:11]{index=11}.
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_exits=3, num_classes=3):
        super(MultiExitResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.exit1 = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.exit2 = nn.Linear(hidden_dim, num_classes)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.exit3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, exit_idx=None):
        x = F.relu(self.fc1(x))
        out1 = self.exit1(x)
        if exit_idx == 1:
            return [out1]
        x2 = F.relu(self.fc2(x)) + x  # residual block
        out2 = self.exit2(x2)
        if exit_idx == 2:
            return [out1, out2]
        x3 = F.relu(self.fc3(x2)) + x2
        out3 = self.exit3(x3)
        return [out1, out2, out3]

def distillation_loss(student_logits, teacher_logits, T=3.0):
    """
    KL-divergence loss for knowledge distillation:contentReference[oaicite:12]{index=12}.
    Students (shallower exits) are trained to match the final exit’s output.
    """
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T*T)
