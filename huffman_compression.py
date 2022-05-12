import argparse
from abc import ABC
from dataclasses import dataclass
from bitarray import bitarray, frozenbitarray
import itertools
import sys
import os

PSEUDO_EOF = bytes()

class HuffmanTree(ABC):
    pass

@dataclass
class Fork(HuffmanTree):
    left: HuffmanTree
    right: HuffmanTree
    weight: int = 0

@dataclass
class Leaf(HuffmanTree):
    data: int | bytes
    weight: int = 0


def weight(tree: HuffmanTree) -> int:
    match tree:
        case Fork(l, r, _):
            return weight(l) + weight(r)
        case Leaf(_, w):
            return w

def build_leaves(freq_stats: dict[int | bytes, int]) -> list[Leaf]:
    return list(map(lambda y: Leaf(y[0], y[1]), sorted(freq_stats.items(), key = lambda x: x[1])))

def concat_trees(left: HuffmanTree, right: HuffmanTree) -> HuffmanTree:
    return Fork(left, right, weight(left) + weight(right))

def combine_once(trees: list[HuffmanTree]) -> list[HuffmanTree]:
    match trees:
        case [t1, t2, *t3]:
            return sorted([concat_trees(t1, t2)] + t3, key = lambda t: weight(t))
        case _:
            return trees

def build_full_tree(freq_stats: dict[int | bytes, int]) -> HuffmanTree:
    result = build_leaves(freq_stats)
    while len(result) > 1:
        result = combine_once(result)
    return result[0]

def make_code(tree: HuffmanTree) -> dict[bytes, bitarray]:
    coding = {}
    def traverse(tree: HuffmanTree, cur_code: bitarray):
        match tree:
            case Fork(l, r, _):
                left_code = cur_code.copy()
                left_code.append(0)
                traverse(l, left_code)
                right_code = cur_code.copy()
                right_code.append(1)
                traverse(r, right_code)
            case Leaf(d, _):
                coding[d] = cur_code
    traverse(tree, bitarray())
    return coding

def encode(source: bytes, coding: dict[bytes, bitarray]) -> bytes:
    result = bitarray()
    for b in source:
        result += coding[b]
    # Don't forget to append pseudo-EOF code
    result += coding[PSEUDO_EOF]
    return result

def decode(source: bitarray, codetree: HuffmanTree) -> bytes:
    coding = make_code(codetree)
    max_codeword_len = max(map(len, coding.values()))
    lookup_table = {}
    for k, v in coding.items():
        augs = list(itertools.product([0, 1], repeat = max_codeword_len - len(v)))
        for a in augs:
            lookup_table[frozenbitarray(v + bitarray(a))] = (k, len(v))
    result = bytes()
    # We intent to retrieve bytes using dict
    # Optimization comes if we know min codeword len
    # Until PSEUDO_EOF encountered
    def get_byte():
        codeword = frozenbitarray(source[:-(max_codeword_len + 1):-1])
        result, offset = lookup_table[codeword]
        del source[-offset:]
        return result

    while True:
        cur_byte = get_byte()
        if cur_byte == PSEUDO_EOF:
            break
        result += cur_byte
    return result

def serialize(tree: HuffmanTree) -> bitarray:
    result = bitarray()
    def traverse(tree: HuffmanTree, tree_code: bitarray):
        match tree:
            case Leaf(d, _):
                tree_code.append(1)
                source_byte = bitarray()
                if d == PSEUDO_EOF:
                    tree_code.append(1)
                else:
                    tree_code.append(0)
                    source_byte.frombytes(bytes([d]))
                tree_code += source_byte
            case Fork(l, r, _):
                tree_code.append(0)
                traverse(l, tree_code)
                traverse(r, tree_code)
    traverse(tree, result)
    return result

def unserialize(tree_code: bitarray) -> HuffmanTree:
    if tree_code.pop() == True:
        eof_bit = tree_code.pop()
        if eof_bit == True:
            # PSEUDO-EOF case
            return Leaf(PSEUDO_EOF)
        leaf_data = bitarray(tree_code[:-9:-1])
        del tree_code[-8:]
        return Leaf(leaf_data.tobytes())
    else:
        left = unserialize(tree_code)
        right = unserialize(tree_code)
        return Fork(left, right)

def compress(source: bytes) -> bytes:
    freq_stats = calc_freq(source)
    coding_tree = build_full_tree(freq_stats)
    coding = make_code(coding_tree)
    serialized_coding = serialize(coding_tree)
    encoded_source = encode(source, coding)
    result = serialized_coding + encoded_source
    return result.tobytes()

def decompress(source: bytes) -> bytes:
    source_bits = bitarray()
    source_bits.frombytes(source)
    # reverse for more efficient pop()'s
    # instead of pop(0)
    source_bits.reverse()
    codetree = unserialize(source_bits)
    result = decode(source_bits, codetree)
    return result

def process_args() -> str:
    parser = argparse.ArgumentParser(description="Huffman coding based compressor")
    parser.add_argument("filename", type=str)
    parser.add_argument("--decompress", action = 'store_true')
    args = parser.parse_args()
    return args.filename, args.decompress

def read_source(filename: str) -> bytes:
    with open(filename, 'rb') as f:
        return f.read()

def calc_freq(source: bytes) -> dict[int | bytes, int]:
    result = dict()
    for b in source:
        if b in result:
            result[b] += 1
        else:
            result[b] = 1
    # Let's solve padding problem using "257-th byte"
    # Bytes of compressed file still can be padded by OS
    # But we will stop earlier, bumping into EOF codeword
    result[PSEUDO_EOF] = 1
    return result


def main():
    filename, decompress_mode = process_args()
    if decompress_mode:
        if filename.endswith('.zmh') and len(os.path.basename(filename)) > 4:
            source = read_source(filename)
            try:
                result = decompress(source)
            except Exception as e:
                print(str(e), file = sys.stderr)
                sys.exit(-1)
            with open(filename[:-4], 'wb') as f:
                f.write(result)
        else:
            print('Wrong file extension, only .zmh files allowed', file = sys.stderr)
            sys.exit(-1)
    else:
        source = read_source(filename)
        result = compress(source)
        with open(f'{filename}.zmh', 'wb') as f:
            f.write(result)

if __name__ == '__main__':
    main()
