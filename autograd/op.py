from enum import Enum

class Operator(Enum):
    """
    An operator in a QAPI expression.
    """
    NONE = ''
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    NEG = '--'
    MATMUL = '@'
    TRANSPOSE = 'T'
    EQ = '=='
    NE = '!='
    RELU = 'ReLU'
