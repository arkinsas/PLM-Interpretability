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


import random


def create_corrupted_swapped(sequence, tensor_idx_i, tensor_idx_j, dataset):
    """
    Replaces the span between contacts with a valid span from a DIFFERENT protein.
    """
    i = tensor_idx_i - 1
    j = tensor_idx_j - 1

    span_start = i + 1
    span_end = j
    span_len = span_end - span_start

    if span_len <= 0:
        return sequence

    # except self
    potential_donors = [seq for name, seq in dataset if seq != sequence]

    if not potential_donors:
        print("Warning: No distinct donors found. Returning original.")
        return sequence

    # grafting
    for _ in range(20):
        donor_seq = random.choice(potential_donors)

        if len(donor_seq) >= span_len:
            max_start = len(donor_seq) - span_len
            start_idx = random.randint(0, max_start)

            grafted_span = donor_seq[start_idx : start_idx + span_len]

            new_sequence = sequence[:span_start] + grafted_span + sequence[span_end:]
            return new_sequence

    return sequence
