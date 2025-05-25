from .ir import Variable, Instruction, DType
from .ir import Compiler, CompilerContext
from .ir import Instruction
from common import Operator


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
        ret_shape = compiler.shapes[compiler.ret]
        ret_note = ' -> memref' + Instruction._mlir_shape(ret_shape)
    else:
        ret_note = ''
    
    arg_list = "("
    args = list(compiler.args)
    for i in range(len(args)):
        arg = args[i]
        if i:
            arg_list += ", "
        arg_list += arg + ": memref" + Instruction._mlir_shape(compiler.shapes[arg])
    arg_list += ")"

    if is_module:
        ir = "module {\n"
        for var in compiler.globl_var:
            shape = compiler.shapes[var]
            ir += f"\t{var} = memref.alloc(): memref" + Instruction._mlir_shape(shape)
            ir += "\n"
        ir += "\tfunc.func @" + funcname + arg_list + ret_note + " {\n"
    else:
        ir = "func.func @" + funcname + arg_list + ret_note + " {\n"

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
