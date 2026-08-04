"""Optional, bounded composition objectives for post-probe training."""

import torch
import torch.nn.functional as F


def text_teacher_kl(student_logits, teacher_logits, mask, temperature=1.0):
    teacher = F.softmax(teacher_logits.float() / temperature, -1)
    student = F.log_softmax(student_logits.float() / temperature, -1)
    values = F.kl_div(student, teacher, reduction="none").sum(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1) * temperature ** 2


def distribution_invariance_kl(clean_logits, perturbed_logits):
    target = F.softmax(clean_logits.detach().float(), -1)
    return F.kl_div(F.log_softmax(perturbed_logits.float(), -1), target,
                    reduction="batchmean")


def slot_cosine_consistency(first, second):
    return (1 - F.cosine_similarity(first.float(), second.float(), dim=-1)).mean()

