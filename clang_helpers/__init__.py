import os
from collections import OrderedDict

import clang
from clang.cindex import CursorKind, TypeKind
import clang.cindex
import platform


# Load the `libclang` included in the `clang-helpers` package for the active
# operating system.
if platform.platform().startswith('Darwin'):
    lib_path = os.path.join(clang.__path__[0], '..', 'libclang', 'libclang.dylib')
    print '[clang-helpers] %s' % lib_path
    clang.cindex.Config.set_library_file(lib_path)
elif platform.platform().startswith('Windows'):
    lib_path = os.path.join(clang.__path__[0], '..', 'libclang', 'libclang.dll')
    print '[clang-helpers] %s' % lib_path
    clang.cindex.Config.set_library_file(lib_path)
else:
    try:
        lib_path = os.path.join(clang.__path__[0], '..', 'libclang',
                                'libclang-3.5.%s.so' % platform.processor())
        clang.cindex.Config.set_library_file(lib_path)
    except:
        from ctypes.util import find_library

        clang.cindex.Config.set_library_file(find_library('clang'))


STD_INT_TYPE = OrderedDict([
    (TypeKind.BOOL, 'bool'),
    (TypeKind.CHAR_S, 'int8_t'),
    (TypeKind.SCHAR, 'int8_t'),
    (TypeKind.CHAR_U, 'uint8_t'),
    (TypeKind.FLOAT, 'float'),
    (TypeKind.INT, 'int32_t'),
    (TypeKind.LONG, 'int32_t'),
    (TypeKind.LONGLONG, 'int64_t'),
    (TypeKind.SHORT, 'int16_t'),
    (TypeKind.UCHAR, 'uint8_t'),
    (TypeKind.UINT, 'uint32_t'),
    (TypeKind.ULONG, 'uint64_t'),
    (TypeKind.USHORT, 'uint16_t'),
    (TypeKind.VOID, None)])


def _get_argument_type(arg):
    if arg.type.kind == TypeKind.POINTER:
        atom_type = arg.type.get_pointee().get_canonical().kind
        return (TypeKind.POINTER, atom_type)
    elif arg.type.kind == TypeKind.RECORD:
        return arg
    else:
        return arg.type.get_canonical().kind


def extract_method_signature(method_cursor):
    definition = method_cursor.get_definition()
    return_type = definition.result_type
    arguments = OrderedDict([(a.displayname, _get_argument_type(a))
                             for a in definition.get_arguments()])
    return (method_cursor.displayname,
            OrderedDict([('return_type', return_type),
                         ('arguments', arguments)]))


def extract_method_signatures(class_cursor):
    method_signatures = [extract_method_signature(m)
                         for m in class_cursor.get_children()
                         if m.kind == CursorKind.CXX_METHOD]

    methods = OrderedDict()

    for s in method_signatures:
        method_name = s[0][:s[0].index('(')]
        signatures = methods.setdefault(method_name, [])
        signatures.append(s[1])
    return methods


def extract_class_declarations(root_cursor):
    return OrderedDict([(n.displayname, n)
                        for n in root_cursor.get_children()
                        if n.kind == CursorKind.CLASS_DECL])


def open_cpp_source(source_path, *args, **kwargs):
    index = clang.cindex.Index.create()
    translational_unit = index.parse(source_path, ['-x', 'c++'] + list(args),
                                     **kwargs)

    return translational_unit.cursor


def get_stdint_type(clang_type_kind):
    try:
        return STD_INT_TYPE[clang_type_kind]
    except TypeError:
        # This is an array type.
        return (STD_INT_TYPE[clang_type_kind['atom_type']], 'array')
