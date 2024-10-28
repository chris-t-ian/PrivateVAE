import hashlib
import numpy as np


def subset_to_sha256_key(subset, len_set=956, len_key=8):
    """Create a unique compact hashed key for a subset of IDs ranging from 0 to len."""
    bit_vector = [0] * len_set
    for num in subset:
        bit_vector[num] = 1

    # Step 2: Convert to a binary string
    binary_str = ''.join(map(str, bit_vector))

    # Step 3: Hash the binary string to get a compact, unique key
    hashed_key = hashlib.sha256(binary_str.encode()).hexdigest()[:len_key]  # Take the first len_key chars for brevity
    return hashed_key


def calculate_AUC(tprs, fprs):
    return np.trapz(tprs, x=fprs)
