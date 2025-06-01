from .ir import Compiler, Variable, Instruction
from fduai.common import Mlir, Operator

class ScopeContext:
    stack = []

class Scope():
    def __init__(self):
        self.children = []

    def __enter__(self):
        if ScopeContext.stack:
            ScopeContext.stack[-1].children.append(self)
        ScopeContext.stack.append(self)
        return self

    def __exit__(self, *args):
        ScopeContext.stack.pop()

class Module(Scope):
    extern_functions = [
        # provided by fduai.compiler.print_tensor.c
        "llvm.func @json_list_start()",
        "llvm.func @json_list_end()",
        "llvm.func @json_list_sep()",
        "llvm.func @json_f32(f32)",
        "llvm.func @new_line()",
    ]

    """
    
    Example:
    >>> from fduai.compiler import Variable, Module, Function
    >>> with Module('main') as m:
    >>>     a = Variable([2, 3])
    >>>     b = Variable([1, 3])
    >>>     with Function('add', a, b) as f:
    >>>         c = a + b
    >>>         f.emit(c)
    """
    def __init__(self, name = None):
        super().__init__()
        self.name = name
        self.compiler = Compiler()

    def __enter__(self):
        super().__enter__()
        self.compiler.__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self.compiler.__exit__(*args)

class Function(Scope):
    """
    
    Example:
    >>> from fduai.compiler import Variable, Function
    >>> a = Variable([2, 3])
    >>> b = Variable([1, 3])
    >>> with Function('add', a, b) as f:
    >>>     c = a + b
    >>>     f.emit(c)
    """
    def __init__(self, name: str, *args):
        super().__init__()
        self.name = name
        self.ret: list[Variable] = []
        self.args: list[Variable] = args

        self.compiler = Compiler()
        for arg in args:
            # self.compiler.add_arg(arg.name)
            self.compiler.vars.append(arg.name)
            self.compiler.shapes[arg.name] = arg.shape
            self.compiler.allocated.add(arg.name)

    def __enter__(self):
        super().__enter__()
        self.compiler.__enter__()

        return self

    def emit(self, ret: Variable):
        self.ret.append(ret)
        self.compiler.ret = [r.name for r in self.ret]

    def emit_many(self, *rets):
        """
        Return multiple return values.

        Example:
        >>> from fduai.compiler import Variable, Function
        >>> a = Variable([2, 3])
        >>> b = Variable([1, 3])
        >>> with Function('add_sub', a, b) as f:
        >>>     c = a + b
        >>>     d = a - b
        >>>     f.emit_many(c, d)
        """
        self.ret = list(rets)
        self.compiler.ret = [r.name for r in self.ret]

    def __exit__(self, *args):
        super().__exit__(*args)
        self.compiler.__exit__(*args)

class Repeat():
    """
    Execute the same code for a number of times.

    Example:
    >>> with Function('main') as f:
    >>>     with Repeat(10):
    >>>         _ = Variable.zeros([1024, 1024])
    >>> print(compile_function(f))
    """
    def __init__(self, times: int):
        self.times = times
        if not ScopeContext.stack:
            raise ValueError("Repeat must be used inside a function")
        self.func: Function = ScopeContext.stack[-1]
        if not isinstance(self.func, Function):
            raise ValueError("Repeat must be used inside a function")

    def __enter__(self):
        self.func.compiler.instructions.append(('for', self.times, [0]))
        return self

    def __exit__(self, *args):
        self.func.compiler.instructions.append(('end_for', 0, [0]))

def compile_function(func: Function, indent: int = 1):
    """
    Compile a function to mlir.
    
    Example:
    >>> from fduai.compiler import Variable, Function
    >>> a = Variable([2, 3])
    >>> b = Variable([1, 3])
    >>> with Function('add', a, b) as f:
    >>>     c = a + b
    >>>     f.emit(c)
    >>> ir = compile_function(f)
    """
    if func.children:
        raise ValueError("Function cannot have children")

    compiler = func.compiler
    rets: list[Variable] = func.ret
    if len(rets):
        shapes: list[str] = []
        for r in rets:
            shapes.append('memref' + Mlir.shape(r.shape))
        ret_note = ' -> ' + (shapes[0] if len(shapes) == 1 else '(' + ', '.join(shapes) + ')')
    else:
        # force main function to return int
        if func.name == 'main':
            ret_note = ' -> i32 '  
        else:
            ret_note = ''
    
    arg_list = "("
    args = func.args
    for i in range(len(args)):
        arg: Variable = args[i]
        if i:
            arg_list += ", "
        arg_list += arg.name + ": memref" + Mlir.shape(arg.shape)
    arg_list += ")"

    ir = '\t' * indent
    ir += "func.func @" + func.name + arg_list + ret_note + " {\n"

    # reserve %zero as index.
    ir += '\t' * indent
    ir += '\t%zero = arith.constant 0 : index\n'

    for insn in compiler.instructions:
        op, output, inputs = insn
        op = Operator.load_from_value(op)

        if op == Operator.END_FOR:
            indent -= 1

        ins = Instruction(op, output, inputs, compiler)
        ir += ins.generate_mlir(indent=indent + 1)
        ir += '\n'

        if op == Operator.FOR:
            indent += 1

    if func.ret:
        insn = Instruction(Operator.RETURN, func.ret[0].name, [], func.compiler)
        ir += insn.generate_mlir(indent=indent + 1)
        ir += '\n'
    else:
        if func.name == "main":
            ir += '\t' * indent
            ir += '\t%main_ret = arith.constant 0 : i32\n'
            ir += '\t' * indent
            ir += "\treturn %main_ret : i32\n"
        else:
            ir += '\t' * indent + '\treturn\n'

    ir += '\t' * indent
    ir += "}\n"

    return ir

def compile_module(module: Module, indent: int = 0) -> str:
    """
    Compile a module to mlir. 

    Example:
    >>> from fduai.compiler import Variable, Module, Function
    >>> with Module('main') as m:
    >>>     a = Variable([2, 3])
    >>>     b = Variable([1, 3])
    >>>     with Function('add', a, b) as f:
    >>>         c = a + b
    >>>         f.emit(c)
    >>> ir = compile_module(m)
    """
    for child in module.children:
        if not isinstance(child, Function):
            raise ValueError("Module must contain only functions")

    ir = '\t' * indent
    ir += 'module '
    if module.name:
        ir += ' @' + module.name
    ir += " {\n"

    for fn in Module.extern_functions:
        ir += '\t' * (indent + 1)
        ir += fn
        ir += '\n'

    for child in module.children:
        ir += compile_function(child, indent=indent + 1)

    ir += '\t' * indent
    ir += '}\n'

    return ir
