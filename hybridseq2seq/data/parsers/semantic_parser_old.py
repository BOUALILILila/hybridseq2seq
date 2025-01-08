from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from collections import defaultdict

from lark import Lark, Token, Tree

import torch


@dataclass
class Node:
    value: str
    position: int = -1

    def __post_init__(self):
        self._is_variable = False
        if self.value.startswith("x _ "):
            self.value = int(self.value[4:])
            self.position = self.position + 4 if self.position > -1 else -1
            self._is_variable = True

    def is_variable(self):
        return self._is_variable

    def is_proper_name(self):
        return (not self.is_variable()) and self.value[0].isupper()

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.position == other.position and self.value == other.value

    def __hash__(self):
        return hash((self.position, self.value))

    def __repr__(self):
        return f"{self.position}: {self.value}"


Dependency = Tuple[Node]
Formula = List[Dependency]


def formula(ast: Tree, add_sos_token: bool) -> Formula:
    if type(ast) == Token:
        if ast.type == "NAME":
            return Node(position=ast.start_pos, value=ast.value.strip())
        elif ast.type == "CONSTANT":
            var = ast.value.strip()
            if add_sos_token and "_" in var:  # the var numbers need to be shifted (+1)
                num = int(var.split("_")[-1]) + 1
                var = f"x _ {num}"
            return Node(position=ast.start_pos, value=var)
        elif ast.type == "START" or ast.type == "END":
            return [(Node(position=ast.start_pos, value=ast.value.strip()),)]
        else:
            raise NotImplementedError

    if ast.data == "ext_formula":
        result = []
        for f in ast.children:
            result.extend(formula(f, add_sos_token))
        return result

    elif ast.data == "formula":
        result = []
        for c in ast.children:
            result.extend(formula(c, add_sos_token))
        return result
    elif ast.data == "predicate":
        return [tuple([formula(c, add_sos_token) for c in ast.children])]
    else:
        raise NotImplementedError


def transform_roles(tokens: List[str], formula: Formula) -> Formula:
    """
    This splits dependency relations in several predicates.
    E.g. eat.theme(x2,x3) => eat(x2) AND theme(x2,x3)
    """
    split_formula = []
    for pred in formula:
        if len(pred) == 3:  # two place predicates
            name, arg1, arg2 = pred
            l = name.value.split(".")
            functor = l[0].strip()
            split_formula.append((Node(value=functor, position=name.position), arg1))
            next_pos = name.position
            for i in range(1, len(l)):
                next_pos += len(l[i - 1]) + 2
                split_formula.append(
                    (Node(value=l[i].strip(), position=next_pos), arg1, arg2)
                )
        elif len(pred) == 2:  # unary predicates
            name, arg1 = pred
            if name.value.startswith("*"):  # definites
                n = name.value[1:].strip()
                arg2 = Node(value=f"x _ {arg1.value - 1}")
                split_formula.append((Node(value="def"), arg1, arg2))
                split_formula.append((Node(value=n, position=name.position + 2), arg1))
                split_formula.append((Node(value="*", position=name.position), arg2))
            else:
                l = name.value.split(".")
                functor = l[0].strip()
                split_formula.append(
                    (Node(value=functor, position=name.position), arg1)
                )
                next_pos = name.position
                for i in range(1, len(l)):
                    next_pos += len(l[i - 1]) + 2
                    split_formula.append(
                        (Node(value=l[i].strip(), position=next_pos), arg1)
                    )
        else:
            (name,) = pred
            if name.value.startswith("*"):  # definites
                n = name.value[1:].strip()
                i = tokens.index(n.lower())
                arg1 = Node(value=f"x _ {i}", position=-1)
                arg2 = Node(value=f"x _ {i - 1}")
                split_formula.append((Node(value="def"), arg1, arg2))
                split_formula.append((Node(value=n, position=name.position + 2), arg1))
                split_formula.append((Node(value="*", position=name.position), arg2))
            else:
                l = name.value.split(".")
                functor = l[0].strip()
                node = Node(value=functor, position=name.position)
                split_formula.append((node,))
                next_pos = name.position
                for i in range(1, len(l)):
                    next_pos += len(l[i - 1]) + 2
                    split_formula.append(
                        (Node(value=l[i].strip(), position=next_pos), node)
                    )
    return list(set(split_formula))


def transform_proper_name_and_word_args(
    tokens: List[Union[str, int]], formula: Formula
) -> Formula:
    """
    Turns proper name constants into unary predicates
    Turns word arguments into unary predicates (assume non duplicates)
    """
    new_preds = set()
    new_formula = []
    for pred in formula:
        new_pred = list(pred)
        if len(pred) == 1:
            elt = pred[0].value
            if elt == "<s>" and elt not in tokens:
                continue
            i = tokens.index(elt.lower())
            var_node = Node(value=f"x _ {i}", position=-1)
            new_preds.add((Node(value=elt, position=pred[0].position), var_node))
        else:
            for idx, elt in enumerate(pred):
                if idx > 0 and (elt.is_proper_name() or (not elt.is_variable())):
                    i = tokens.index(elt.value.lower())
                    var_node = Node(value=f"x _ {i}", position=-1)
                    new_preds.add((elt, var_node))
                    new_pred[idx] = var_node
            new_formula.append(tuple(new_pred))
    return list(set(new_formula + list(new_preds)))


def add_functionals(tokens: List[str], formula: Formula) -> Formula:
    """
    Adds the 'other' predicates
    """
    assigned_vars = set(pred[-1].value for pred in formula if len(pred) == 2)
    for idx, elt in enumerate(tokens):
        if idx not in assigned_vars:
            formula.append((Node(value="other"), Node(value=f"x _ {idx}")))
    return formula


def apply_transforms(tokens: List[str], formula: Formula) -> Formula:
    formula = transform_roles(tokens, formula)
    formula = transform_proper_name_and_word_args(tokens, formula)
    return formula


def map_char_to_token_pos(sent):
    char_to_token_map = []
    token_pos = 0
    for char in sent:
        if char != " ":
            char_to_token_map.append(token_pos)
        else:
            char_to_token_map.append(-1)
            token_pos += 1
    return char_to_token_map


class DependencyGraph:
    def __init__(
        self,
        relation_keywords: List,
        token_pos_map: List,
        relation_edge_weight: float = 0.5,
    ) -> None:
        self._graph = nx.Graph()
        self._relation_edge = relation_edge_weight
        self._token_pos_map = token_pos_map
        self._relation_keywords = relation_keywords
        self._map = defaultdict(lambda: -100)

    def add_dependency(self, dep: Dependency):
        if len(dep) == 3:  # Relation
            self.add_edge(dep[0], dep[1], weight=self._relation_edge)
            self.add_edge(dep[0], dep[2], weight=self._relation_edge)
        elif len(dep) == 2:
            # Avoid incomplete relations
            # if dep[0].value not in self._relation_keywords:
            #     self.add_edge(dep[0], dep[1], weight = 0)
            if dep[0].value in self._relation_keywords:
                self.add_edge(dep[0], dep[1], weight=self._relation_edge)
            else:
                self.add_edge(dep[0], dep[1], weight=0)

    # def add_node(self, node):
    #     self.graph.add_node(node.position)
    #     return node.position

    def add_edge(self, node_1: Node, node_2: Node, weight: float = 1):
        if node_1.position >= 0:
            node_1_pos = self._token_pos_map[node_1.position]
            self._map[node_1_pos] = node_1.value
        if node_2.position >= 0:
            node_2_pos = self._token_pos_map[node_2.position]
            self._map[node_2_pos] = node_2.value
        self._graph.add_edge(node_1.value, node_2.value, weight=weight)

    def num_nodes(self):
        return len(self._graph)

    def plot_dependency_tree(self):
        plt.figure(1, figsize=(10, 10))
        g = self._graph
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, with_labels=True, node_size=1600, font_size=10)
        labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.show()

    def _catch_no_path(self, func, default: float, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except nx.NetworkXNoPath:
            return default

    def get_distance_matrix(self, n: int):
        # Distance Matrix
        D = torch.zeros((n, n))
        for idx in range(n):
            # Djikstra shortest path ==> lower triangular distance matrix
            p = [
                self._catch_no_path(
                    nx.shortest_path_length,
                    0.0,
                    self._graph,
                    source=self._map[idx],
                    target=self._map[tgt_id],
                    weight="weight",
                )
                if (self._map[idx] in self._graph and self._map[tgt_id] in self._graph)
                else 0.0
                for tgt_id in range(idx)
            ]
            D[idx, :idx] = torch.tensor(p)
        return D + D.T  # symmetric distance matrix


class SemanticParser:
    """Builds a tree structure of words from the input sentence, which represents the syntactic dependency relations between words."""

    def __init__(
        self,
        epsilon: float,
        relation_edge_weight: float = 0.5,
        pad_token: str = "<pad>",
        add_sos_token: Optional[bool] = False,
    ) -> None:
        self._relation_keywords = {
            "agent",
            "theme",
            "recipient",
            "xcomp",
            "ccomp",
            "nmod",
        }
        self._stop_words = {
            ";",
            "(",
            ")",
            "AND",
            "and",
            ".",
            "X",
            "x",
            "_",
            ",",
            pad_token,
        }
        if not add_sos_token:
            self._stop_words.add("<s>")
        self._stop_words_ext = self._stop_words.union({"*"})
        self.epsilon = epsilon
        self.relation_edge_weight = relation_edge_weight
        self.add_sos_token = add_sos_token

        cogs_grammar = r"""
        ext_formula : START (formula (";" formula)*)? END?
        formula     : predicate ( ("and"|"AND") predicate)*
        predicate   : NAME  ("(" CONSTANT ("," CONSTANT)* ")"? )?

        START     : "<s>"
        END       : "</s>"
        NAME      : /(\* )?[A-Za-z]+( \. [A-Za-z]+)*/
        CONSTANT  : /[A-Za-z]+( *_ *[0-9]+)?/

        %import common.WS
        %ignore WS
        """
        self.cogs_parser = Lark(
            cogs_grammar, parser="lalr", start="ext_formula", debug=True
        )

    def _parse_logic_to_tuples(
        self, src_tokens: List[int], previous_target_tokens: str
    ):
        ast = self.cogs_parser.parse(previous_target_tokens)
        return apply_transforms(src_tokens, formula(ast, self.add_sos_token))

    def _get_relation_distance_from_closest_arg(self, d_arg: torch.Tensor):
        return d_arg + self.epsilon + self.relation_edge_weight

    def _get_node_distance_from_corresp(self, d_arg: torch.Tensor):
        return d_arg + self.epsilon

    def get_distance_matrix(
        self,
        step: int,
        previous_cross_distance_matrix: torch.Tensor,  # (bs, step-1, step-1)
        source_distance_matrix: torch.Tensor,  # (bs, src_seq_length, src_seq_length)
        source_tokens_list: List[List[str]],  # (bs, src_seq_length)
        previous_target_tokens_list: List[List[str]],  # (bs, step)
        target_attention_mask_step: torch.Tensor,  # (bs, step)
    ):
        # Shape , device
        # target mask in both cases

        src_seq_length = source_distance_matrix.shape[1]
        new_shape = (
            source_distance_matrix.shape[0],  # bs
            src_seq_length + step,
            src_seq_length + step,
        )
        cross_distance_matrix = torch.zeros(
            new_shape,
            device=source_distance_matrix.device,
            dtype=source_distance_matrix.dtype,
        )

        ZERO = torch.zeros_like(target_attention_mask_step[0, 0])

        for i, (source_tokens, previous_target_tokens) in enumerate(
            zip(source_tokens_list, previous_target_tokens_list)
        ):
            if (
                previous_target_tokens[-1] in self._stop_words_ext
            ):  # or (previous_target_tokens[-1] not in source_tokens)
                # if previous_target_tokens[-1] not in source_tokens:
                #     print("###########")
                #     print(f"ingoring {previous_target_tokens[-1]} token since it is not in the source {source_tokens}")
                #     print("###########")
                target_attention_mask_step[i, -1] = ZERO
                cross_distance_matrix[
                    i, : src_seq_length + step - 1, : src_seq_length + step - 1
                ] = previous_cross_distance_matrix[i]
            else:
                target_attention_mask_step[i] = torch.tensor(
                    [
                        0.0 if tok in self._stop_words else 1.0
                        for tok in previous_target_tokens
                    ],
                    device=target_attention_mask_step.device,
                    dtype=target_attention_mask_step.dtype,
                )
                previous_target_seq = " ".join(previous_target_tokens)
                cross_distance_matrix[i, :src_seq_length, :src_seq_length] = (
                    source_distance_matrix[i].clone().detach()
                )
                try:
                    dependency_tuples = self.get_dependency_tuples(
                        source_tokens, previous_target_seq
                    )
                except Exception as e:
                    print(e)
                    print("\n\n=======================")
                    print("source = ", source_tokens)
                    print("tgt = ", previous_target_seq)
                    print("=======================\n\n")
                    exit(0)

                char_to_token_pos = map_char_to_token_pos(previous_target_seq)

                # cross distances + target semantic graph construction
                target_dep_graph = DependencyGraph(
                    self._relation_keywords,
                    char_to_token_pos,
                    self.relation_edge_weight,
                )
                for dep in dependency_tuples:
                    target_dep_graph.add_dependency(dep)
                    if dep[0].value == "def":
                        continue
                    if (
                        dep[0].value in self._relation_keywords
                    ):  # relation (no equivalent in source sequence)
                        idx = char_to_token_pos[dep[0].position] + src_seq_length
                        closest_orig_token_distances = [
                            self._get_relation_distance_from_closest_arg(
                                cross_distance_matrix[i, arg.value, :src_seq_length]
                            )
                            for arg in dep[1:]
                        ]
                        d_rel = torch.stack(closest_orig_token_distances).min(-2).values
                        cross_distance_matrix[i, idx, :src_seq_length] = d_rel
                        cross_distance_matrix[
                            i, :src_seq_length, idx
                        ] = d_rel  # symmetric
                    else:  # original words
                        idx = char_to_token_pos[dep[0].position] + src_seq_length
                        d_node = self._get_node_distance_from_corresp(
                            cross_distance_matrix[i, dep[1].value, :src_seq_length]
                        )
                        # word
                        cross_distance_matrix[i, idx, :src_seq_length] = d_node
                        cross_distance_matrix[
                            i, :src_seq_length, idx
                        ] = d_node  # symmetric
                        # its variable
                        if dep[1].position > -1:  # variable in target
                            idx = char_to_token_pos[dep[1].position] + src_seq_length
                            cross_distance_matrix[i, idx, :src_seq_length] = d_node
                            cross_distance_matrix[
                                i, :src_seq_length, idx
                            ] = d_node  # symmetric

                # self distances
                target_distance_matrix = target_dep_graph.get_distance_matrix(
                    len(previous_target_tokens)
                )  # step = len(previous_...)
                cross_distance_matrix[
                    i, src_seq_length:, src_seq_length:
                ] = target_distance_matrix
                assert torch.allclose(
                    cross_distance_matrix[i],
                    cross_distance_matrix[i].T,
                    rtol=1e-5,
                    atol=1e-8,
                ), "not symmetric"
        return cross_distance_matrix, target_attention_mask_step

    def get_dependency_tuples(
        self,
        source_tokens: torch.Tensor,
        previous_target_tokens: torch.Tensor,
    ) -> Formula:
        return self._parse_logic_to_tuples(source_tokens, previous_target_tokens)
