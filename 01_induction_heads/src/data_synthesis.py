import random
from utils import SEED, CONTEXT_LENGTH, VOCAB_SIZE

random.seed(SEED)

def generate_random_token(length: int, vocab_size: int = VOCAB_SIZE, repeated: bool = True) -> list:
    """
    Generate a random sequence of tokens from the vocabulary list.

    Args:
        length (int): The length of the sequence to generate.

    Returns:
        list: A list of tokens representing the generated sequence.
    """
    if repeated:
        return [random.randint(0, vocab_size - 1) for _ in range(length)]
    else:
        return random.sample(range(vocab_size), length)

def generate_sequence_parts(
        seq_length: int = CONTEXT_LENGTH,
        max_prefix: int = 8,
        min_pattern: int = 6,
        max_pattern: int = 12,
        min_gap: int = 2,
        max_gap: int = 16,
        repeated: bool = True) -> tuple:
    """
    Generate a random prefix, pattern, gap, and suffix.
    """
    prefix_length = random.randint(0, max_prefix)
    pattern_length = random.randint(min_pattern, max_pattern)
    gap_length = random.randint(min_gap, max_gap)
    suffix_length = seq_length - (prefix_length + pattern_length + gap_length + pattern_length)

    prefix = generate_random_token(prefix_length)
    pattern = generate_random_token(pattern_length, repeated=repeated)
    gap = generate_random_token(gap_length)
    suffix = generate_random_token(suffix_length)
    
    return prefix, pattern, gap, suffix

def generate_sequence(
        seq_length: int = CONTEXT_LENGTH,
        max_prefix: int = 8,
        min_pattern: int = 6,
        max_pattern: int = 12,
        min_gap: int = 2,
        max_gap: int = 16,
        repeated: bool = True) -> list:
    """
    Generate a sequence with a random prefix, pattern, gap, and suffix.
    """
    prefix, pattern, gap, suffix = generate_sequence_parts(seq_length, max_prefix, min_pattern, max_pattern, min_gap, max_gap, repeated)
    return prefix + pattern + gap + pattern + suffix
