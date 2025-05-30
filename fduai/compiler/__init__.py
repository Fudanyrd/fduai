from .ir import Variable, Instruction, DType
from .ir import Compiler, CompilerContext
from .ir import Instruction
from .scope import Function, Module, compile_function, compile_module
from fduai.common import Operator
from fduai.common import Mlir


import json


def generate_mlir(compiler=None, funcname: str = 'start', 
                  is_module: bool = False, indent: int = 0) -> str:
    """
    :param compiler: Compiler instance to compile
    :param indent: Indentation level
    :param funcname: Name of the function to compile
    :param is_module: Whether the function is a module
    :return: Compiled IR
    """
    if compiler is None:
        compiler = CompilerContext.compiler

    if compiler is None:
        raise RuntimeError('No compiler found')

    if compiler.ret:
        ret_shape = compiler.retshape()
        ret_note = ' -> ' + Mlir.rettype(ret_shape)
    else:
        ret_note = ''
    
    arg_list = "("
    args = list(compiler.globl_var) + list(compiler.args)
    for i in range(len(args)):
        arg = args[i]
        if i:
            arg_list += ",\n\t\t"
        arg_list += arg + ": memref" + Instruction._mlir_shape(compiler.shapes[arg])
    arg_list += ")"

    if is_module:
        ir = "module {\n\t"
    else:
        ir = ""
    ir += "func.func @" + funcname + arg_list + ret_note + " {\n"
    # reserve %zero for index
    if is_module:
        ir += '\t'
    ir += '\t%zero = arith.constant 0 : index\n'

    for ins in compiler.instructions:
        op, output, inputs = ins
        op = Operator.load_from_value(op)

        ins = Instruction(op, output, inputs, compiler)
        ir += ins.generate_mlir(indent=indent + (2 if is_module else 1))
        ir += '\n'

    if is_module:
        ir += "\t}\n}\n"
    else:
        ir += "}\n"
    return ir
