from __future__ import print_function

from __future__ import absolute_import
from collections import OrderedDict
import sys

from clang.cindex import CursorKind, TypeKind
import clang
import clang.cindex
import path_helpers as ph

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


STD_INT_TYPE = OrderedDict([
    (TypeKind.BOOL, 'bool'),
    (TypeKind.CHAR_S, 'int8_t'),
    (TypeKind.SCHAR, 'int8_t'),
    (TypeKind.CHAR_U, 'uint8_t'),
    (TypeKind.FLOAT, 'float'),
    (TypeKind.DOUBLE, 'double'),
    (TypeKind.INT, 'int32_t'),
    (TypeKind.LONG, 'int32_t'),
    (TypeKind.LONGLONG, 'int64_t'),
    (TypeKind.SHORT, 'int16_t'),
    (TypeKind.UCHAR, 'uint8_t'),
    (TypeKind.UINT, 'uint32_t'),
    (TypeKind.ULONG, 'uint64_t'),
    (TypeKind.USHORT, 'uint16_t'),
    (TypeKind.VOID, None)])


STD_INT_KIND = OrderedDict([
    ('BOOL', 'bool'),
    ('CHAR_S', 'int8_t'),
    ('SCHAR', 'int8_t'),
    ('CHAR_U', 'uint8_t'),
    ('FLOAT', 'float'),
    ('DOUBLE', 'double'),
    ('INT', 'int32_t'),
    ('LONG', 'int32_t'),
    ('LONGLONG', 'int64_t'),
    ('SHORT', 'int16_t'),
    ('UCHAR', 'uint8_t'),
    ('UINT', 'uint32_t'),
    ('ULONG', 'uint64_t'),
    ('USHORT', 'uint16_t'),
    ('VOID', None),
])


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


def extract_class_declarations(root_cursor, include_templates=True):
    def _extract_classes_level(root_cursor, include_templates=True):
        return OrderedDict([(n.displayname, n)
                            for n in root_cursor.get_children()
                            if (n.kind == CursorKind.CLASS_DECL) or
                            (include_templates and
                            (n.kind == CursorKind.CLASS_TEMPLATE))])

    def _extract_classes(root_cursor, classes, parents=None,
                         include_templates=True):
        classes_ = iter(_extract_classes_level(root_cursor).items())
        if parents is None:
            parents = tuple()
        classes.update(OrderedDict([('::'.join([p.displayname
                                                for p in parents] + [k]), v)
                                    for k, v in classes_]))
        # Recurse through namespaces to find available classes.
        for c in root_cursor.get_children():
            if c.kind == CursorKind.NAMESPACE:
                _extract_classes(c, classes, parents + (c, ))

    classes = OrderedDict()
    _extract_classes(root_cursor, classes, include_templates=include_templates)
    return classes


def open_cpp_source(source_path, *args, **kwargs):
    '''
    .. versionchanged:: 0.9
        Add header directory from ``clang-libcxx`` Conda package to includes
        path, if it is available (Windows only).  Note that the ``CPATH``
        environment varaible can also be modified to add additional directories
        to the header search path.
    '''
    index = clang.cindex.Index.create()
    default_args = ['-x', 'c++']
    if hasattr(sys, 'prefix'):
        clang_libcxx_dir = ph.path(sys.prefix).joinpath('Library', 'include',
                                                        'clang-libcxx')
        if clang_libcxx_dir.isdir():
            default_args += ['-I', clang_libcxx_dir]
    translational_unit = index.parse(source_path, default_args + list(args),
                                     **kwargs)

    return translational_unit.cursor


def get_stdint_type(clang_type_kind):
    try:
        return STD_INT_TYPE[clang_type_kind]
    except TypeError:
        # This is an array type.
        return (STD_INT_TYPE[clang_type_kind['atom_type']], 'array')
