import random

def create_corrupted_shuffled(sequence, tensor_idx_i, tensor_idx_j):
    """
    Shuffles the span between contacts to break the interaction context.
    """
    i = tensor_idx_i - 1
    j = tensor_idx_j - 1

    span_start = i + 1
    span_end = j

    if span_end <= span_start:
        return sequence

    seq_list = list(sequence)
    middle_segment = seq_list[span_start:span_end]
    random.shuffle(middle_segment)
    seq_list[span_start:span_end] = middle_segment

    return "".join(seq_list)