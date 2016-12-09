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


def mergedicts(dict1, dict2):
    # See [here][1].
    #
    # [1]: http://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge/7205107#7205107
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


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


def extract_base_identifiers(class_node):
    '''
    Extract list of ``(IDENTIFIER, specifier_dict)`` pairs, where each
    ``IDENTIFIER`` is a token in the base specifier list of the specified
    class.

    Note
    ----
        Extracted list may contain ``IDENTIFIER`` tokens that are **not** class
        names, e.g., preprocessor ``ifdef``, ``endif``  tokens.

    Parameters
    ----------
    class_node : clang.cindex.Cursor
        clang cursor referencing a C++ class declaration.

    Returns
    -------
    list
        List of ``IDENTIFIER`` tokens in base specifier list.
    '''
    processing = False

    template_stack = []
    base_specifiers = []
    previous_token = None

    for t in class_node.get_tokens():
        if t.spelling == ':':
            processing = True
        if t.spelling == '<':
            template_stack.append(t)
        if t.spelling == '>':
            template_stack.pop()
        if processing and not template_stack and t.kind.name == 'IDENTIFIER':
            node_obj = {'token': t}
            if previous_token.spelling in ('public', 'private', 'protected'):
                node_obj['access_specifier'] = getattr(clang.cindex
                                                       .AccessSpecifier,
                                                       previous_token.spelling
                                                       .upper())
            else:
                node_obj['access_specifier'] = 'PROTECTED'
            base_specifiers.append((t.spelling, node_obj))
        if t.spelling == '{':
            break
        previous_token = t

    return base_specifiers


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
    result = {'underlying_type': type_i,
              'type': type_,
              'kind': type_i.kind,
              'const': type_.is_const_qualified(),
              'volatile': type_.is_volatile_qualified(),
              'restrict': type_.is_restrict_qualified()}

    if type_i.kind.name == 'POINTER':
        result['pointer'] = True
        type_i = type_i.get_pointee()
        result['const'] = type_i.is_const_qualified()
        result['underlying_type'] = resolve_typedef(type_i)
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
    def __init__(self, add_index=False):
        '''
        Parameters
        ----------
        add_index : bool, optional
            If ``True``, add an ``'index'`` property to each node in the AST
            according to the order the nodes are processed in the parsing walk.
        '''
        super(CppAst, self).__init__()
        self._access_specifier = None
        self._in_class = False
        self._in_method = False
        self._in_function = False
        self._in_typedef = False
        self._parents = []
        self.root = OrderedDict()
        self._debug_output = False
        self.count = 0
        self.add_index = add_index

    def get_class(self, class_name):
        '''
        Look up class object by name, e.g., `foo::bar::FooBar`.
        '''
        f_get_class = get_class_factory(self.root['translation_units']
                                        .values()[0])
        return f_get_class(class_name)

    def _class_obj(self, class_name):
        class_ = self.get_class(class_name)
        if class_ is None:
            return {}
        else:
            node = class_['node']
            return {'class_node': node, 'type': node.type}

    def _extract_base_specifiers(self, class_node):
        '''
        Construct base specifier objects from extracted identifier tokens in
        base specifier list of specified class.

        Parameters
        ----------
        class_node : clang.cindex.Cursor
            Cursor to class node in clang AST.

        Returns
        -------
        OrderedDict
            Mapping from class names to corresponding base specifier objects.
        '''
        base_identifiers = extract_base_identifiers(class_node)
        return OrderedDict(filter(lambda v: 'class_node' in v[1],
                           [(name, dict(mergedicts(b, self._class_obj(name))))
                            for name, b in base_identifiers]))

    def _merge_base_specifiers(self, class_, inplace=False):
        '''
        Parameters
        ----------
        class_ : dict
            Class object, possibly containing an item with the key
            ``'base_specifiers'``.
        inplace : bool, optional
            If ``True``, directly modify ``'base_specifiers'`` item on
            :data:`class_`.

        Returns
        -------
        None or OrderedDict
            If :data:`inplace` is set to ``True``, return ``None``.

            Otherwise, return base specifiers extracted from tokens merged with
            with base specifiers from clang AST.
        '''
        token_base_specifiers = self._extract_base_specifiers(class_['node'])
        base_specifiers = class_.get('base_specifiers', OrderedDict())

        items = []
        token_base_iter = token_base_specifiers.iteritems()

        for i, (name_i, base_i) in enumerate(base_specifiers.iteritems()):
            for name_j, base_j in token_base_iter:
                if name_j != name_i:
                    items.append((name_j, base_j))
                else:
                    break
            items.append((name_i, base_i))

        result = OrderedDict(items)
        if inplace:
            class_['base_specifiers'] = result
        else:
            return result

    def _merge_all_base_specifiers(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.iteritems():
                if k == 'classes':
                    for name_i, class_i in v.iteritems():
                        self._merge_base_specifiers(class_i, inplace=True)
                else:
                    self._merge_all_base_specifiers(v)
        elif isinstance(obj, list):
            for v in obj:
                self._merge_all_base_specifiers(v)

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
        if node.kind is clang.cindex.CursorKind.TYPEDEF_DECL:
            self._in_typedef = False
            self._active_typedef = None
        if self._debug_output:
            print node.spelling,
            super(CppAst, self).leaveNode(node, level)

    def visitNode(self, node, level):
        self.count += 1
        # super(CppAst, self).visitNode(node, level)
        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE,
                         clang.cindex.CursorKind.STRUCT_DECL):
            self._in_class = True
        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE):
            self._access_specifier = 'PROTECTED'
        elif node.kind is clang.cindex.CursorKind.STRUCT_DECL:
            self._access_specifier = 'PUBLIC'
        if node.kind is clang.cindex.CursorKind.FUNCTION_DECL:
            self._in_function = True

        parents = node_parents(node)
        if self._debug_output:
            print self.count, '*' * level, node.spelling,
            super(CppAst, self).visitNode(node, level)
            print format_parents(parents),
            print '{}<{}> ({}, line={} col={})'.format(node.spelling,
                                                       trimClangNodeName(node.kind),
                                                       None if not
                                                       node.location.file else
                                                       node.location.file.name,
                                                       node.location.line,
                                                       node.location.column)

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
        if self.add_index:
            node_obj['count'] = self.count

        if node.kind is clang.cindex.CursorKind.TRANSLATION_UNIT:
            translation_units_i = parent.setdefault('translation_units',
                                                    OrderedDict())
            translation_units_i[node.displayname] = node_obj
            return

        parent_objs = []
        for parent_node_i in parents:
            if parent_node_i.kind is clang.cindex.CursorKind.TRANSLATION_UNIT:
                # Top-level parent.
                parent = parent['translation_units'][parent_node_i.displayname]
            elif parent_node_i.kind in (clang.cindex.CursorKind.CLASS_DECL,
                                        clang.cindex.CursorKind.CLASS_TEMPLATE,
                                        clang.cindex.CursorKind.STRUCT_DECL):
                if parent_node_i.spelling:
                    parent = parent['classes'][parent_node_i.spelling]
                else:
                    parent = parent['anonymous_classes'][-1]
            elif parent_node_i.kind in (clang.cindex.CursorKind.NAMESPACE, ):
                parent = parent['namespaces'][parent_node_i.spelling]
            parent_objs.append(parent)

        if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                         clang.cindex.CursorKind.CLASS_TEMPLATE,
                         clang.cindex.CursorKind.STRUCT_DECL):
            if self._in_typedef:
                # Processing `typedef struct`, so use `typedef` name instead.
                name = self._active_typedef.spelling
            else:
                name = node.spelling
            if name:
                classes_i = parent.setdefault('classes', OrderedDict())
                classes_i[name] = node_obj
                self._class = node_obj
            else:
                classes_i = parent.setdefault('anonymous_classes', [])
                classes_i.append(node_obj)
                self._class = node_obj
            return

        if node.kind is clang.cindex.CursorKind.NAMESPACE:
            namespaces_i = parent.setdefault('namespaces', OrderedDict())
            # Merge with existing namespaces object, since the same namespace
            # may span multiple files.
            node_obj = dict(mergedicts(namespaces_i.get(node.spelling, {}),
                                       node_obj))
            namespaces_i[node.spelling] = node_obj
            return

        if node.kind is clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
            template_types_i = parent.setdefault('template_types',
                                                  OrderedDict())
            template_types_i[node.displayname] = node_obj
            return

        if node.kind is clang.cindex.CursorKind.TYPEDEF_DECL:
            self._in_typedef = True
            self._active_typedef = node
            typedefs_i = parent.setdefault('typedefs', OrderedDict())

            type_i = resolve_typedef(node)
            node_obj.update({'type': type_i, 'kind': type_i.kind})
            typedefs_i[node.spelling] = node_obj
            return

        # # Process members #
        node_obj['name'] = node.spelling
        if self._in_class:
            node_obj['access_specifier'] = self._access_specifier
            parent = self._class
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
        elif node.kind in (clang.cindex.CursorKind.CXX_METHOD,
                           clang.cindex.CursorKind.FUNCTION_DECL):
            if node.kind is clang.cindex.CursorKind.CXX_METHOD:
                self._in_method = True
            members_i = parent.setdefault('members', OrderedDict())

            result_type = resolve_typedef(node.result_type)
            result_kind = STD_INT_KIND.get(result_type.kind.name,
                                           result_type.kind.name)

            args = [{'name': a.displayname, 'node': a}
                    for a in node.get_arguments()]

            for arg_i in args:
                arg_i.update(type_node(arg_i['node'].type))

            if result_kind == 'UNEXPOSED':
                if result_type.spelling in parent.get('template_types', {}):
                    result_kind = '::'.join([parents[-1].displayname,
                                             result_type.spelling])
            elif result_kind == 'RECORD':
                result_kind = result_type.spelling
            node_obj.update({'result_type': result_kind,
                             'kind': node.kind,
                             'arguments': args})

            if node.is_definition():
                for child_i in node.get_children():
                    if child_i.kind is clang.cindex.CursorKind.COMPOUND_STMT:
                        tokens = list(child_i.get_tokens())
                        # Token 0 is the opening bracket of the compound
                        # statement.
                        # If the next token is a comment, store the comment as
                        # a description of the function.
                        if tokens[1].kind is clang.cindex.TokenKind.COMMENT:
                            node_obj['description'] = tokens[1].spelling
                        break
            members_i[node.spelling] = node_obj


def parse_cpp_ast(input_file, *args, **kwargs):
    '''
    Parameters
    ----------
    input_file : str
        Input file path.
    add_index : bool, optional
        If ``True``, add an ``'index'`` property to each node in the AST
        according to the order the nodes are processed in the parsing walk.
    extract_base_specifiers : bool, optional
        If ``True``, extract base specifiers for all classes from identifier
        tokens in base specifier list.
    format : str, optional
        If `'clang'`, maintain clang instances for types, cursors, etc.  The
        resulting abstract syntax tree (AST) dictionary **cannot be
        serialized**.

        If `'json'`, convert instances of clang types to JSON-serializable
        format (i.e., strings).
    *args : optional
        Additional arguments to pass to :meth:`clang.cindex.Index.parse`.
    **kwargs : optional
        Additional keyword arguments to pass to
        :meth:`clang.cindex.Index.parse`.

    Returns
    -------
    dict
        Abstract syntax tree (AST) dictionary in format specified by
        :data:`format` parameter (i.e., either `clang` or `json` format).
    '''
    add_index = kwargs.pop('add_index', False)
    extract_base_specifiers = kwargs.pop('extract_base_specifiers', False)
    format_ = kwargs.pop('format', 'clang')
    # If the line below fails , set Clang library path with
    # clang.cindex.Config.set_library_path
    clang_index = clang.cindex.Index.create()
    translationUnit = clang_index.parse(input_file, ['-x', 'c++'] + list(args),
                                        **kwargs)
    root_node = translationUnit.cursor

    cpp_ast = CppAst(add_index=add_index)
    cpp_ast.walkAST(root_node, 0)

    root = cpp_ast.root['translation_units'].values()[0]
    if extract_base_specifiers:
        cpp_ast._merge_all_base_specifiers(root)
    if format_ == 'json':
        _format_json_safe(root)
    return root


def _format_json_safe(obj):
    '''
    Remove/replace ctype instances, leaving only values that are
    json-serializable.
    '''
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (clang.cindex.TypeKind, clang.cindex.CursorKind)):
                obj[k] = v.name
            elif isinstance(v, clang.cindex.AccessSpecifier):
                obj[k] = v.name
            elif isinstance(v, clang.cindex.Cursor):
                f_extent = lambda marker: dict([(attr, getattr(marker, attr))
                                               for attr in ['line',
                                                            'column']])
                obj['location'] = dict(zip(['file', 'start', 'end'],
                                            (v.location.file.name
                                             if v.location.file else None,
                                             f_extent(v.extent.start),
                                             f_extent(v.extent.end))))
                obj['name'] = v.spelling
                del obj[k]
            elif isinstance(v, clang.cindex.Type):
                obj[k] = v.spelling
            elif isinstance(v, (dict, list)):
                _format_json_safe(v)
        if 'type' in obj and 'kind' in obj:
            obj['type'] = STD_INT_KIND.get(obj['kind'], obj['type'])
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


def public_members(cpp_ast_json, class_name):
    '''
    Parameters
    ----------
    cpp_ast_json : dict
        Root of json-formatted C++ abstract syntax tree.
    class_name : str
        Class name, e.g., `foo::bar::FooBar`.

    Returns
    -------
    list
        List of publicly exposed members from specified class and all base
        classes, sorted by member name.

    Notes
    -----
        Multiple methods with the same signature are overridden based on order
        of inheritance.

        Behaviour for overloaded methods (i.e., multiple methods with different
        call signatures) is currently **undefined**.
    '''
    get_class_json = get_class_factory(cpp_ast_json)
    f_public_members = (py_.curry_right(py_.pick_by)
                        (lambda v, k: (v.get('access_specifier') in
                                       ('PUBLIC', None))))

    def _public_members(class_name, members):
        class_ = get_class_json(class_name)
        class_members = [dict(mergedicts({'class': class_name, 'name': name_i},
                                         method_i))
                         for name_i, method_i in
                         f_public_members(class_.get('members', {}))
                         .iteritems()]
        members.extend(sorted(class_members, key=lambda v: v['name']))

        for name_i, base_i in reversed(class_.get('base_specifiers', {})
                                       .items()):
             if base_i.get('access_specifier') in (None, 'PUBLIC'):
                _public_members(name_i, members)
    members = []
    _public_members(class_name, members)
    return sorted(py_.uniq(members, 'name'), key=lambda v: v['name'])


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

    cpp_ast_json = parse_cpp_ast(header_file, *include_args, format='json')
    cpp_ast = parse_cpp_ast(header_file, *include_args)

    get_class = get_class_factory(cpp_ast)
    get_class_json = get_class_factory(cpp_ast_json)
