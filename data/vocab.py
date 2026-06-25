from collections import OrderedDict
from typing import List

VOCAB_MISSING = OrderedDict([
    ('t',-1),   # no data bc disorderd terminus
    ('x',-1),   # no data, R1/R2/NOE not reported
    ('p',0),    # proline, not evaluated
    ('A', 0),   # nothing
    ('v',0),    # fast motion
    ('.',1),    # missing
    ('b',0),    # both fast and slow
    ('^',0)     # rex
])

VOCAB_REX = OrderedDict([
    ('t',-1),   # no data bc disorderd terminus
    ('x',-1),   # no data, R1/R2/NOE not reported
    ('p',0),    # proline, not evaluated
    ('A', 0),   # nothing
    ('v',0),    # fast motion
    ('.',1),    # missing
    ('b',1),    # both fast and slow
    ('^',1)     # rex
])

VOCAB_CPMG = OrderedDict([
    ('t',-1),   # no data bc disordered terminus
    ('P',0),    # proline, not evaluated
    ('N',-1),   # no data, assned but CPMG not reported
    ('A', 0),   # nothing
    ('.',1),    # missing
    ('X',1),    # exchange from Rex definition
    ('Y',0)     # exchange from unsuppressed R2
])

def mask_termini(seq):
    """
    Mask the termini of a sequence
    """
    seq = seq.lstrip('.p').rjust(len(seq), 't')
    seq = seq.rstrip('.p').ljust(len(seq), 't')
    return seq

class label_tokenizer():
    def __init__(self, 
                 type = 'missing', 
                 missing_only = False, 
                 rex_only = False, 
                 unsuppressed = False):
        """
        Tokenize the data labeling for BMRB, REX, and CPMG
        
        Args:
            type: (str) which type of experiment
            missing_only: (bool) only return residues with missing peaks
            rex_only: (bool) only return residues with Rex
            unsuppressed: (bool) return residues with unsuppressed Rex
        """
        if type == 'missing':
            self.vocab = VOCAB_MISSING.copy()
        elif type == 'rex':
            self.vocab = VOCAB_REX.copy()
            if missing_only:
                self.vocab['b'] = 0
                self.vocab['^'] = 0
        elif type == 'cpmg':
            self.vocab = VOCAB_CPMG.copy()
            if missing_only:
                self.vocab['X'] = 0
            if unsuppressed:
                self.vocab['Y'] = 1
        if rex_only:
            self.vocab['.'] = -1
        self.tokens = list(self.vocab.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str], pad_to_length = None) -> List[int]:
        """Converts a list of tokens (str/unicode) into a list of ids (int) using the vocab. """

        if pad_to_length is None:
          return [self.convert_token_to_id(token) for token in tokens]
        else:
          return [self.convert_token_to_id(token) for token in tokens] + [0] * (pad_to_length - len(tokens))

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        """Converts a list of indices (integer) into a list of tokens (string/unicode) using the vocab."""
        return [self.convert_id_to_token(id_) for id_ in indices]
