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
        }

        return d[value]
