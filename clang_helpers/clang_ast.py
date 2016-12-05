#!/usr/bin/env python
#
#
# Understanding CLang AST : http://clang.llvm.org/docs/IntroductionToTheClangAST.html
#
# Requirements
#
# 1. Install Clang :  Visit http://llvm.org/releases/download.html
#
# 2. pip install clang
#
# Note : Make sure that your Python runtime and Clang are same architecture
#
#Core
from collections import OrderedDict
import sys

import clang
import clang.cindex
import path_helpers as ph
import pydash as py_
from . import STD_INT_KIND


class DotOrderedDict(OrderedDict):
    def __getattr__(self, attr):
        try:
            result = super(DotOrderedDict, self).__getattribute__(attr)
            return result
        except AttributeError:
            if attr == '_OrderedDict__root':
                raise
            return self.get(attr, None)

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super(DotOrderedDict, self).__setattr__(attr, value)
        else:
            self.__setitem__(attr, value)

    __delattr__ = OrderedDict.__delitem__


class CppAstWalker(object):
    @staticmethod
    def trimClangNodeName(nodeName):
        ret = str(nodeName)
        ret = ret.split(".")[1]
        return ret

    @staticmethod
    def printASTNode(node, level, exit=False):
        for i in range(0, level):
            print '  ',
        if exit is True:
            print ("Exiting " + CppAstWalker.trimClangNodeName(node.kind))
        else:
            print CppAstWalker.trimClangNodeName(node.kind)

    def visitNode(self, node, level):
        CppAstWalker.printASTNode(node, level)

    def leaveNode(self, node, level):
        CppAstWalker.printASTNode(node, level, True)

    def walkAST(self, node, level):
        if node is not None:
            level = level + 1
            self.visitNode(node, level)
        # Recurse for children of this node
        for childNode in node.get_children():
            self.walkAST(childNode, level)
        self.leaveNode(node, level)
        level = level - 1


def node_parents(node):
    parents = []

    node_i = node

    while node_i.lexical_parent:
        node_i = node_i.lexical_parent
        parents.insert(0, node_i)
    return parents


def format_parents(parents):
    return '::'.join(['{}<{}>'.format(p_i.displayname,
                                      str(p_i.kind).split('.')[-1])
                      for p_i in parents])


def resolve_typedef(typedef_node):
    '''
    Parameters
    ----------
    typedef_node : clang.cindex.Cursor

    Returns
    -------
    clang.cindex.Type
        Underlying type from ``typedef``.

        If input :data:`typedef_node` was not a type definition, return
        :data:`typedef_node`.
    '''
    if typedef_node.kind is clang.cindex.TypeKind.TYPEDEF:
        typedef_node = typedef_node.get_declaration()
    if not typedef_node.kind is clang.cindex.CursorKind.TYPEDEF_DECL:
        return typedef_node

    #Find underlying type of (possibly nested) typedef.
    type_i = typedef_node.underlying_typedef_type
    while type_i.kind == clang.cindex.TypeKind.TYPEDEF:
        type_i = type_i.get_declaration().underlying_typedef_type
    return type_i


def type_node(type_):
    type_i = resolve_typedef(type_)
    result = {'type': type_i,
              'typename': type_i.spelling,
              'kind': type_i.kind,
              'const': type_.is_const_qualified()}

    if type_i.kind.name == 'POINTER':
        result['pointer'] = True
        type_i = type_i.get_pointee()
        result['const'] = type_i.is_const_qualified()
        result['type'] = resolve_typedef(type_i)
        result['kind'] = result['type'].kind
    elif type_i.kind.name in ('CONSTANTARRAY', 'INCOMPLETEARRAY'):
        # Extract resolved element type and size from constant-sized array.
        result['element_type'] = resolve_typedef(type_i
                                                 .get_array_element_type())
        result['element_kind'] = result['element_type'].kind
        array_size = type_i.get_array_size()
        if array_size >= 0:
            result['array_size'] = array_size
    return result


def trimClangNodeName(nodeName):
    ret = str(nodeName)
    ret = ret.split(".")[1]
    return ret


class CppAst(CppAstWalker):
    def __init__(self):
        super(CppAst, self).__init__()
        self._access_specifier = None
        self._in_class = False
        self._in_method = False
        self._in_function = False
        self._parents = []
        self.root = OrderedDict()

    def leaveNode(self, node, level):
        # super(CppAst, self).leaveNode(node, level)
        if node.kind is clang.cindex.CursorKind.CXX_METHOD:
            self._in_method = False
        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE,
                         clang.cindex.CursorKind.STRUCT_DECL):
            self._in_class = False
            self._access_specifier = 'PROTECTED'
        if node.kind is clang.cindex.CursorKind.FUNCTION_DECL:
            self._in_function = False

    def visitNode(self, node, level):
        # super(CppAst, self).visitNode(node, level)
        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE,
                         clang.cindex.CursorKind.STRUCT_DECL):
            self._access_specifier = 'PROTECTED'
            self._in_class = True
        if node.kind is clang.cindex.CursorKind.FUNCTION_DECL:
            self._in_function = True

        parents = node_parents(node)
        # print format_parents(parents),
        # print '{}<{}> (line={} col={})'.format(name,
                                               # trimClangNodeName(node.kind),
                                               # node.location.line,
                                               # node.location.column)

        if node.kind is clang.cindex.CursorKind.CXX_ACCESS_SPEC_DECL:
            self._access_specifier = node.access_specifier
            return

        if node.kind is clang.cindex.CursorKind.CXX_BASE_SPECIFIER:
            base_specifiers_i = self._class.setdefault('base_specifiers',
                                                       OrderedDict())
            base_specifiers_i[node.type.spelling] = {'node': node,
                                                     'type': node.type,
                                                     'access_specifier':
                                                     node.access_specifier}
            return

        parent = self.root

        node_obj = {'node': node}

        if node.kind is clang.cindex.CursorKind.TRANSLATION_UNIT:
            translation_units_i = parent.setdefault('translation_units',
                                                    OrderedDict())
            translation_units_i[node.displayname] = node_obj
            return

        for parent_node_i in parents:
            if parent_node_i.kind is clang.cindex.CursorKind.TRANSLATION_UNIT:
                # Top-level parent.
                parent = parent['translation_units'][parent_node_i.displayname]
            elif parent_node_i.kind in (clang.cindex.CursorKind.CLASS_DECL,
                                        clang.cindex.CursorKind.CLASS_TEMPLATE,
                                        clang.cindex.CursorKind.STRUCT_DECL):
                parent = parent['classes'][parent_node_i.spelling]
            elif parent_node_i.kind in (clang.cindex.CursorKind.NAMESPACE, ):
                parent = parent['namespaces'][parent_node_i.spelling]

        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE,
                         clang.cindex.CursorKind.STRUCT_DECL):
            classes_i = parent.setdefault('classes', OrderedDict())
            classes_i[node.spelling] = node_obj
            self._class = classes_i[node.spelling]
            return

        if node.kind is clang.cindex.CursorKind.NAMESPACE:
            namespaces_i = parent.setdefault('namespaces', OrderedDict())
            namespaces_i[node.spelling] = node_obj
            return

        if node.kind is clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
            template_types_i = parent.setdefault('template_types',
                                                  OrderedDict())
            template_types_i[node.displayname] = node_obj
            return

        if node.kind is clang.cindex.CursorKind.TYPEDEF_DECL:
            typedefs_i = parent.setdefault('typedefs', OrderedDict())

            type_i = resolve_typedef(node)
            node_obj.update({'type': type_i, 'kind': type_i.kind})
            typedefs_i[node.spelling] = node_obj
            return

        # # Process members #
        node_obj['name'] = node.spelling
        if self._in_class:
            node_obj['access_specifier'] = self._access_specifier
        if ((node.kind is clang.cindex.CursorKind.VAR_DECL) and
            (not self._in_function and (not self._in_class or not
                                        self._in_method))):
            members_i = parent.setdefault('members', OrderedDict())
            node_obj.update(type_node(node.type))
            members_i[node.spelling] = node_obj
        elif ((node.kind is clang.cindex.CursorKind.FIELD_DECL) and
              (not self._in_function and (not self._in_class or not
                                          self._in_method))):
            members_i = parent.setdefault('members', OrderedDict())
            node_obj.update(type_node(node.type))
            members_i[node.spelling] = node_obj
        elif node.kind is clang.cindex.CursorKind.CXX_METHOD:
            self._in_method = True
            members_i = parent.setdefault('members', OrderedDict())

            result_type = resolve_typedef(node.result_type)
            result_kind = result_type.kind.name

            args = [{'name': a.displayname, 'node': a}
                    for a in node.get_arguments()]

            for arg_i in args:
                arg_i.update(type_node(arg_i['node'].type))

            if result_kind == 'UNEXPOSED':
                if result_type.spelling in parent.get('template_types', {}):
                    result_kind = '::'.join([parents[-1].displayname,
                                             result_type.spelling])
            node_obj.update({'result_type': result_kind,
                             'kind': node.type.kind,
                             'typename': 'method',
                             'arguments': args})

            if node.is_definition():
                comments_i = []
                for child_i in node.get_children():
                    if child_i.kind is clang.cindex.CursorKind.COMPOUND_STMT:
                        comments_i = [t for t in child_i.get_tokens()
                                      if t.kind is
                                      clang.cindex.TokenKind.COMMENT]
                        break

                if comments_i:
                    node_obj['description'] = comments_i[0].spelling
            members_i[node.spelling] = node_obj


def parse_cpp_ast(input_file, *args, **kwargs):
    '''
    Parameters
    ----------
    input_file : str
        Input file path.
    *args : optional
        Additional arguments to pass to :meth:`clang.cindex.Index.parse`.
    **kwargs : optional
        Additional keyword arguments to pass to
        :meth:`clang.cindex.Index.parse`.
    '''
    # If the line below fails , set Clang library path with
    # clang.cindex.Config.set_library_path
    clang_index = clang.cindex.Index.create()
    translationUnit = clang_index.parse(input_file, ['-x', 'c++'] + list(args),
                                        **kwargs)
    root_node = translationUnit.cursor

    cpp_ast = CppAst()
    cpp_ast.walkAST(root_node, 0)
    return cpp_ast.root['translation_units'].values()[0]


def _format_json_safe(obj):
    '''
    Remove/replace ctype instances, leaving only values that are
    json-serializable.
    '''
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, clang.cindex.TypeKind):
                obj[k] = STD_INT_KIND.get(v.name, v.name)
            elif isinstance(v, clang.cindex.AccessSpecifier):
                obj[k] = v.name
            elif isinstance(v, clang.cindex.Cursor):
                obj['location'] = dict(zip(['file', 'line', 'column'],
                                            (v.location.file.name
                                             if v.location.file else None,
                                             v.location.line,
                                             v.location.column)))
                obj['name'] = v.spelling
                del obj[k]
            elif isinstance(v, clang.cindex.Type):
                obj[k] = v.spelling
            elif isinstance(v, (dict, list)):
                _format_json_safe(v)
    elif isinstance(obj, list):
        remove_indexes = []
        for i, v in enumerate(obj):
            if isinstance(v, clang.cindex.Cursor):
                remove_indexes.append(i)
            elif isinstance(v, (dict, list)):
                _format_json_safe(v)
        for i in remove_indexes:
            del obj[i]


def show_location(location, stream=sys.stdout):
    with open(location['file'], 'r') as source:
        line = source.readlines()[location['line']]
        print >> stream, '# {file} line {line} col {column}'.format(**location)
        print >> stream, ''
        print >> stream, line.strip()
        print >> stream, ' ' * (location['column'] - 1) + '^'


def get_class_path(class_str):
    parts_i = class_str.split('::')
    return ('namespaces.' + '.namespaces.'.join(parts_i[:-1]) + '.'
            if parts_i[:-1] else '') + 'classes.' + parts_i[-1]


# Generate function to look up class by name, e.g., `foo::bar::FooBar`.
get_class_factory = lambda ast: py_.pipe(get_class_path,
                                         py_.curry(py_.get, arity=2)(ast))


def parse_args(args=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if args is None:
        args = sys.argv

    parser = ArgumentParser(description='Parse clang AST to nested dictionary')
    parser.add_argument('input_file', type=ph.path)
    parser.add_argument('-I', '--include', type=ph.path, action='append')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    include_args = ['-I{}'.format(p) for p in args.include]
    header_file = ph.path(sys.argv[1]).realpath()

    cpp_ast_json = parse_cpp_ast(header_file, *include_args)
    _format_json_safe(cpp_ast_json)
    cpp_ast = parse_cpp_ast(header_file, *include_args)

    get_class = get_class_factory(cpp_ast)
    get_class_json = get_class_factory(cpp_ast_json)
