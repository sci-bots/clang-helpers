from collections import OrderedDict

from ctypes.util import find_library
from clang.cindex import CursorKind
import clang.cindex

clang.cindex.Config.set_library_file(find_library('clang'))


def extract_method_signature(method_cursor):
    definition = method_cursor.get_definition()
    return_type = definition.result_type.get_canonical().kind
    arguments = OrderedDict([(a.displayname, a.type.get_canonical().kind)
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
