
class Mlir():
    """
    See also:  
    memref dialect: https://mlir.llvm.org/docs/Dialects/MemRef/  
    arith dialect: https://mlir.llvm.org/docs/Dialects/Arith/  
    mlir language reference: https://mlir.llvm.org/docs/LangRef/  
    """
    @staticmethod 
    def shape(shape, dtype='f32') -> str:
        return '<' + 'x'.join([str(x) for x in shape]) + f'x{dtype}>'

    @staticmethod
    def index(output_shape, input_shape) -> str:
        indices = ['%zero' if input_shape[i] == 1 else f'%arg{i}' for i in range(len(output_shape))]
        return '[' + ', '.join(indices) + ']'

    @staticmethod 
    def typename(shape, dtype='f32'):
        return 'memref' + Mlir.shape(shape, dtype)

    @staticmethod 
    def rettype(shapes):
        return '(' + ', '.join([Mlir.typename(shape) for shape in shapes]) + ')'

    @staticmethod
    def retstmt(vars, shapes):
        # %r1, %r2 : memref<1xf32>, memref<1xf32>
        assert len(vars) == len(shapes)
        return ', '.join(vars) + " : " + ', '.join([Mlir.typename(shape) for shape in shapes])
