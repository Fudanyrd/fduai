"""
fduai.op
===========
We use a simple IR for this project. Our IR supports the following instructions:

**ADD**: Add two tensors.
**SUB**: Subtract two tensors.
**MUL**: Multiply two tensors.
**DIV**: Divide two tensors.
**NEG**: Negate a tensor.
**MATMUL**: Matrix multiplication.
**TRANSPOSE**: Transpose a tensor.
**EQ**: Equal.
**NE**: Not equal.
**LT**: Less than.
**GT**: Greater than.
**RELU**: ReLU activation function.
**PRINT**: Print a tensor in JSON format.
**MOV**: Move a tensor.
"""
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
    LT = '<'
    GT = '>'
    RELU = 'ReLU'
    RESHAPE = 'reshape'
    RETURN = 'return'
    INIT = 'init'
    FILL = 'fill'
    FOR = 'for'
    END_FOR = 'end_for'
    PRINT = 'print'
    MOV = 'move'


    @staticmethod
    def load_from_value(value):
        d = {
            "": Operator.NONE,
            "+": Operator.ADD,
            "-": Operator.SUB,
            "*": Operator.MUL,
            "/": Operator.DIV,
            "--": Operator.NEG,
            '@': Operator.MATMUL,
            'T': Operator.TRANSPOSE,
            '==': Operator.EQ,
            '!=': Operator.NE,
            '<': Operator.LT,
            '>':  Operator.GT,
            'ReLU': Operator.RELU,
            'reshape': Operator.RESHAPE,
            "return": Operator.RETURN,
            "init": Operator.INIT,
            "fill": Operator.FILL,
            "for": Operator.FOR,
            'end_for': Operator.END_FOR,
            'print': Operator.PRINT,
            'move': Operator.MOV,
        }

        return d[value]
