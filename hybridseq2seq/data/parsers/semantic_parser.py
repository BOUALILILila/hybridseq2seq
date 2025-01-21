from collections import defaultdict
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
from lark import Discard, Lark, Token, Transformer, v_args
from lark.parsers.earley_forest import TreeForestTransformer
from networkx.drawing.nx_pydot import graphviz_layout


class Node:
    def __init__(self, value: str, position: int, is_relation: bool = False):
        self._is_variable = False
        self._is_relation = is_relation
        self.value = (
            value.lower()
        )  # all text is lower case / uncase mode is used for parsing
        self.position = position
        if self.value.startswith("x _ "):
            self.value = int(self.value[4:])
            self.position = self.position + 2 if self.position > -1 else -1
            self._is_variable = True
        if self._is_relation:  # Identify relation keywords with name+position
            self.value = f"{self.value}.{self.position}"

    def is_variable(self):
        return self._is_variable

    def is_relation(self):
        return self._is_relation

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.position == other.position and self.value == other.value

    def __hash__(self):
        return hash((self.position, self.value))

    def __repr__(self):
        return f"Node(position={self.position}, value={self.value})"


class DependencyGraph:
    def __init__(
        self, relation_edge_weight: float = 0.5, default_max_distance: int = 0.0
    ) -> None:
        self._graph = nx.Graph()
        self._relation_edge = relation_edge_weight
        self.default_max_distance = default_max_distance
        self._map = defaultdict(lambda: -1)

    def add_edge(self, node_1: Node, node_2: Node, weight: float = 1):
        if node_1.position >= 0:
            self._map[node_1.position] = node_1.value
        if node_2.position >= 0:
            self._map[node_2.position] = node_2.value
        self._graph.add_edge(node_1.value, node_2.value, weight=weight)

    def add_node(self, node: Node):
        if node.position >= 0:
            self._map[node.position] = node.value
        self._graph.add_node(node.value)

    def num_nodes(self):
        return len(self._graph)

    def plot_dependency_tree(self):
        plt.figure(1, figsize=(5, 5))
        g = self._graph
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, with_labels=True, node_size=1000, font_size=10)
        labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.show()

    def _catch_no_path(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except nx.NetworkXNoPath:
            return self.default_max_distance

    def get_distance_matrix(self, n: int):
        # Distance Matrix
        D = torch.zeros((n, n))
        for idx in range(n):
            # Djikstra shortest path ==> lower triangular distance matrix
            p = [
                (
                    self._catch_no_path(
                        nx.shortest_path_length,
                        self._graph,
                        source=self._map[idx],
                        target=self._map[tgt_id],
                        weight="weight",
                    )
                    if (
                        self._map[idx] in self._graph
                        and self._map[tgt_id] in self._graph
                    )
                    else self.default_max_distance
                )
                for tgt_id in range(idx)
            ]
            D[idx, :idx] = torch.tensor(p)
        return D + D.T  # symmetric distance matrix


def map_char_to_token_pos(sent: str):
    char_to_token_map = []
    token_pos = 0
    for char in sent:
        if char != " ":
            if len(char_to_token_map) > 0 and char_to_token_map[-1] == -1:
                token_pos += 1
            char_to_token_map.append(token_pos)
        else:
            char_to_token_map.append(-1)
    return char_to_token_map


@v_args(inline=True)  # Affects the signatures of the methods
class AbstractDependencyTreeTransformer(Transformer):
    """Visits the abstract tree of the target sequence and extracts (1) the dependency
    graph, and (2) the cross dependencies between the target-source tokens.
    """

    def __init__(
        self,
        src_tokens: List[str],
        tgt_seq: str,
        add_sos_token: bool = False,
        relation_edge_weight: float = 0.5,
        default_max_distance: float = 30,
    ):
        self.char2token_pos_map = map_char_to_token_pos(tgt_seq)
        self.relation_edge_weight = relation_edge_weight
        self.dependency_graph = DependencyGraph(
            default_max_distance=default_max_distance
        )
        self.add_sos_token = add_sos_token
        self.src_tokens = src_tokens
        self.cross_dependency_tuples = set()  # tgt-src cross dependencies
        self.cross_ref = defaultdict(lambda: -1)  # cross target-source map

    def _map_word_to_src(self, w: str) -> int:
        if w in self.src_tokens:
            i = self.src_tokens.index(w)
            self.cross_ref[w] = i
            return i
        return -1

    def _map_var_to_src(self, val: int) -> int:
        if val < len(self.src_tokens):
            self.cross_ref[val] = val
            return val
        return -1

    def START(self, token: Token):
        # node = Node(value=token.value.strip(), position=self.char2token_pos_map[token.start_pos])
        # self.dependency_graph.add_node(node)
        # return node
        return None

    def END(self, token: Token):
        # node = Node(value=token.value.strip(), position=self.char2token_pos_map[token.start_pos])
        # self.dependency_graph.add_node(node)
        # return node
        return None

    def STOP(self, token: Token):
        return None

    def MSTOP(self, token: Token):
        return None

    def WORD(self, token: Token) -> Node:
        word_node = Node(
            value=token.value.strip(), position=self.char2token_pos_map[token.start_pos]
        )
        self.dependency_graph.add_node(word_node)
        i = self._map_word_to_src(word_node.value)
        self.cross_dependency_tuples.add(
            (word_node.position, i)
        )  # source map dependency
        return word_node

    def VAR(self, token: Token) -> Node:
        val = int(token.value.strip().split("_")[-1])
        val = val + 1 if self.add_sos_token else val
        node = Node(
            value=f"x _ {val}", position=self.char2token_pos_map[token.start_pos]
        )
        self.dependency_graph.add_node(node)
        self.cross_dependency_tuples.add(
            (node.position, self._map_var_to_src(val))
        )  # source map dependency
        return node

    def REL(self, token: Token) -> Node:
        node = Node(
            value=token.value.strip(),
            position=self.char2token_pos_map[token.start_pos],
            is_relation=True,
        )
        self.dependency_graph.add_node(node)
        return node

    def NMOD(self, token: Token) -> Node:
        node = Node(
            value=token.value.strip(),
            position=self.char2token_pos_map[token.start_pos],
            is_relation=True,
        )
        self.dependency_graph.add_node(node)
        return node

    def PREP(self, token: Token) -> Node:
        node = Node(
            value=token.value.strip(),
            position=self.char2token_pos_map[token.start_pos],
            is_relation=True,
        )
        self.dependency_graph.add_node(node)
        return node

    def indefinite(self, word: Node, var: Node) -> None:
        self.dependency_graph.add_edge(word, var, weight=0.0)  # predicted dependency

    def definite(self, word: Node, var: Node) -> None:
        # dep = (word, var)
        # self.dependency_tuples.add(dep)
        # def_article = Node(value = f'x _ {var.value - 1}') # node for "the" = "*"
        # var_node = Node(value = "*", position = word.position-2)
        # self.dependency_tuples.append(
        #     (def_article, var_node)
        # )
        self.dependency_graph.add_edge(word, var, weight=0.0)

    def binary_relation(
        self, word: Node, relation: Node, var: Node, constant: Node
    ) -> None:
        self.dependency_graph.add_edge(word, var, weight=0.0)
        self.dependency_graph.add_edge(relation, var, weight=self.relation_edge_weight)
        self.dependency_graph.add_edge(
            relation, constant, weight=self.relation_edge_weight
        )

        self.cross_dependency_tuples.add(
            (
                relation.position,
                self.cross_ref[var.value],
                self.cross_ref[constant.value],
            )
        )

    def nmod_relation(
        self, word: Node, relation: Node, prep: Node, var: Node, constant: Node
    ) -> None:
        self.dependency_graph.add_edge(word, var, weight=0.0)
        self.dependency_graph.add_edge(constant, prep)
        self.dependency_graph.add_edge(relation, var, weight=self.relation_edge_weight)
        self.dependency_graph.add_edge(
            relation, constant, weight=self.relation_edge_weight
        )

        # nmod node
        self.cross_dependency_tuples.add(
            (
                relation.position,
                self.cross_ref[var.value],
                self.cross_ref[constant.value],
            )
        )
        # preposition node
        i = self.cross_ref[constant.value] - 2
        self.cross_dependency_tuples.add((prep.position, i))  # source map dependency

    def var(self, node: Node) -> Node:
        return node

    def constant(self, node: Node) -> Node:
        return node

    def __default__(self, data, children, meta):
        return children

    def WS(self, token: Token):
        return Discard

    def __default_token__(self, token: Token):
        return Discard


class SemanticParser:
    """Builds a tree structure of words from the input sentence, which represents the
    semantic dependency relations between words."""

    def __init__(
        self,
        epsilon: float,
        relation_edge_weight: float = 0.5,
        default_max_distance: float = 30,
        pad_token: str = "<pad>",
        sos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
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
            "AND",
            "and",
            ".",
            "X",
            "x",
            "_",
            ",",
            "*",
            "LAMBDA",
            pad_token,
            eos_token,
            unk_token,
        }  # ")",
        if not add_sos_token:
            self._stop_words.add(sos_token)
        self._stop_words_ext = self._stop_words.union({")"})
        self.epsilon = epsilon
        self.relation_edge_weight = relation_edge_weight
        self.add_sos_token = add_sos_token
        self.default_max_distance = default_max_distance

        cogs_grammar = r"""
                ext_formula     : START (WS formula)? (WS END)?

                formula         : predicate
                                | formula WS predicate
                
                predicate       : relation
                                | constant
                                | STOP
                                | MSTOP
                                | "and"
                                | "<s>"
                                | "<unk>"
                                | "<pad>"
                
                relation.1      : definite
                                | indefinite
                                | WORD WS "." WS REL WS "(" WS VAR WS "," WS constant WS ")"             -> binary_relation
                                | WORD WS "." WS NMOD WS "." WS PREP WS "(" WS VAR WS "," WS constant WS ")"   -> nmod_relation
                
                definite.1      : "*" WS WORD WS "(" WS VAR WS ")"  
                indefinite      : WORD WS "(" WS VAR WS ")"

                constant        : VAR
                                | WORD

                REL       : "agent" | "theme" | "recipient" | "xcomp" | "ccomp"
                NMOD      : "nmod"
                PREP      : /[A-Za-z]+/
                START     : "<s>"
                END.1     : "</s>"
                VAR       : /x _ [0-9]+/
                WORD      : /[A-Za-z]+/
                STOP      : /[^A-Za-z\s]/
                MSTOP     : /[^A-Za-z\s]+/
                WS        : " "
        """
        self.cogs_parser = Lark(
            cogs_grammar, start="ext_formula", parser="earley", ambiguity="forest"
        )

    def get_dependency_tuples(
        self,
        source_tokens: List[int],
        previous_target_tokens: str,
    ) -> Tuple[DependencyGraph, Set[Tuple[int, int]]]:
        # parse the target seq
        forest = self.cogs_parser.parse(previous_target_tokens)
        # resolve ambiguity
        tree = TreeForestTransformer(resolve_ambiguity=True).transform(forest)
        # build the dependency graph and target-source cross references
        dep_transform = AbstractDependencyTreeTransformer(
            src_tokens=source_tokens,
            tgt_seq=previous_target_tokens,
            add_sos_token=self.add_sos_token,
            relation_edge_weight=self.relation_edge_weight,
            default_max_distance=self.default_max_distance,
        )
        dep_transform.transform(tree)
        return dep_transform.dependency_graph, dep_transform.cross_dependency_tuples

    def _get_relation_distance_from_closest_arg(
        self, cross_distance_matrix: torch.Tensor, idx: int
    ) -> torch.Tensor:
        if idx > -1:
            return cross_distance_matrix[idx] + self.epsilon + self.relation_edge_weight
        return torch.ones_like(cross_distance_matrix[0]) * self.default_max_distance

    def _get_node_distance_from_corresp(
        self, cross_distance_matrix: torch.Tensor, idx: int
    ) -> torch.Tensor:
        if idx > -1:
            return cross_distance_matrix[idx] + self.epsilon
        return torch.ones_like(cross_distance_matrix[0]) * self.default_max_distance

    def get_distance_matrix(
        self,
        step: int,
        previous_cross_distance_matrix: torch.Tensor,  # (bs, step-1, step-1)
        source_distance_matrix: torch.Tensor,  # (bs, src_seq_length, src_seq_length)
        source_tokens_list: List[List[str]],  # (bs, src_seq_length)
        previous_target_tokens_list: List[List[str]],  # (bs, step)
        target_attention_mask_step: torch.Tensor,  # (bs, step)
    ):
        src_seq_length = source_distance_matrix.shape[1]
        new_shape = (
            source_distance_matrix.shape[0],  # bs
            src_seq_length + step,
            src_seq_length + step,
        )
        # init the cross_distance_matrix
        cross_distance_matrix = torch.zeros(
            new_shape,
            device=source_distance_matrix.device,
            dtype=source_distance_matrix.dtype,
        )
        # for masking
        ZERO = torch.zeros_like(target_attention_mask_step[0, 0])

        for i, (source_tokens, previous_target_tokens) in enumerate(
            zip(source_tokens_list, previous_target_tokens_list)
        ):
            if (
                previous_target_tokens[-1] in self._stop_words
            ):  # tokens that do not affect the previous dependencies
                target_attention_mask_step[i, -1] = ZERO
                cross_distance_matrix[
                    i, : src_seq_length + step - 1, : src_seq_length + step - 1
                ] = previous_cross_distance_matrix[i]
            else:
                target_attention_mask_step[i] = torch.tensor(
                    [
                        0.0 if tok in self._stop_words_ext else 1.0
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
                    (
                        target_dependency_graph,
                        cross_dependencies,
                    ) = self.get_dependency_tuples(source_tokens, previous_target_seq)
                except Exception as e:
                    print(e)
                    print("\n\n=======================")
                    print("source = ", source_tokens)
                    print("tgt = ", previous_target_seq)
                    print("=======================\n\n")
                    exit(0)

                # cross distances computation
                for dep in cross_dependencies:
                    if len(dep) == 3:  # (relation, var, constant)
                        idx = dep[0] + src_seq_length
                        closest_orig_token_distances = [
                            self._get_relation_distance_from_closest_arg(
                                cross_distance_matrix[i, :, :src_seq_length], arg
                            )
                            for arg in dep[1:]
                        ]
                        d_rel = torch.stack(closest_orig_token_distances).min(-2).values
                        cross_distance_matrix[i, idx, :src_seq_length] = d_rel
                        cross_distance_matrix[i, :src_seq_length, idx] = (
                            d_rel  # symmetric
                        )
                    elif len(dep) == 2:  # original words or variables
                        idx = dep[0] + src_seq_length
                        d_node = self._get_node_distance_from_corresp(
                            cross_distance_matrix[i, :, :src_seq_length], dep[1]
                        )
                        cross_distance_matrix[i, idx, :src_seq_length] = d_node
                        cross_distance_matrix[i, :src_seq_length, idx] = (
                            d_node  # symmetric
                        )
                    else:
                        raise ValueError(
                            f"Dependency {dep} cannot be trated. Length of a dependency must be of 2 or 3."
                        )

                # tgt distances
                target_distance_matrix = target_dependency_graph.get_distance_matrix(
                    len(previous_target_tokens)
                )  # step = len(previous_...)
                cross_distance_matrix[i, src_seq_length:, src_seq_length:] = (
                    target_distance_matrix
                )
                assert torch.allclose(
                    cross_distance_matrix[i],
                    cross_distance_matrix[i].T,
                    rtol=1e-5,
                    atol=1e-8,
                ), "not symmetric"
        return cross_distance_matrix, target_attention_mask_step

    def get_one_distance_matrix(
        self,
        step: int,
        previous_cross_distance_matrix: torch.Tensor,  # (step-1, step-1)
        source_distance_matrix: torch.Tensor,  # (src_seq_length, src_seq_length)
        source_tokens: List[List[str]],  # (src_seq_length)
        previous_target_tokens: List[List[str]],  # (step)
        target_attention_mask_step: torch.Tensor,  # (step)
        ended: Optional[torch.Tensor] = None,  # tensor(True/False)
    ):
        src_seq_length = source_distance_matrix.shape[-1]
        new_shape = (
            src_seq_length + step,
            src_seq_length + step,
        )
        # init the cross_distance_matrix
        cross_distance_matrix = torch.zeros(
            new_shape,
            device=source_distance_matrix.device,
            dtype=source_distance_matrix.dtype,
        )
        # for masking
        ZERO = torch.zeros_like(target_attention_mask_step[0])

        if not (
            ended or previous_target_tokens[-1] in self._stop_words
        ):  # no (stop words or generated eos token)
            target_attention_mask_step = torch.tensor(
                [
                    0.0 if tok in self._stop_words_ext else 1.0
                    for tok in previous_target_tokens
                ],
                device=target_attention_mask_step.device,
                dtype=target_attention_mask_step.dtype,
            )
            previous_target_seq = " ".join(previous_target_tokens)
            cross_distance_matrix[:src_seq_length, :src_seq_length] = (
                source_distance_matrix.clone().detach()
            )
            try:
                (
                    target_dependency_graph,
                    cross_dependencies,
                ) = self.get_dependency_tuples(source_tokens, previous_target_seq)
            except Exception as e:
                print("\n\n=======================")
                print("parsing exception:")
                print("ended = ", ended)
                print(e)
                print("source = ", source_tokens)
                print("tgt = ", previous_target_seq)
                print("=======================\n\n")
                exit(0)

            # cross distances computation
            for dep in cross_dependencies:
                if len(dep) == 3:  # (relation, var, constant)
                    idx = dep[0] + src_seq_length
                    closest_orig_token_distances = [
                        self._get_relation_distance_from_closest_arg(
                            cross_distance_matrix[:, :src_seq_length], arg
                        )
                        for arg in dep[1:]
                    ]
                    d_rel = torch.stack(closest_orig_token_distances).min(-2).values
                    cross_distance_matrix[idx, :src_seq_length] = d_rel
                    cross_distance_matrix[:src_seq_length, idx] = d_rel  # symmetric
                elif len(dep) == 2:  # original words or variables
                    idx = dep[0] + src_seq_length
                    d_node = self._get_node_distance_from_corresp(
                        cross_distance_matrix[:, :src_seq_length], dep[1]
                    )
                    cross_distance_matrix[idx, :src_seq_length] = d_node
                    cross_distance_matrix[:src_seq_length, idx] = d_node  # symmetric
                else:
                    raise ValueError(
                        f"Dependency {dep} cannot be trated. Length of a dependency must be of 2 or 3."
                    )

            # tgt distances
            target_distance_matrix = target_dependency_graph.get_distance_matrix(
                len(previous_target_tokens)
            )  # step = len(previous_...)
            cross_distance_matrix[src_seq_length:, src_seq_length:] = (
                target_distance_matrix
            )
            assert torch.allclose(
                cross_distance_matrix,
                cross_distance_matrix.T,
                rtol=1e-5,
                atol=1e-8,
            ), "not symmetric"

        else:
            target_attention_mask_step[-1] = ZERO
            cross_distance_matrix[
                : src_seq_length + step - 1, : src_seq_length + step - 1
            ] = previous_cross_distance_matrix

        return cross_distance_matrix, target_attention_mask_step


if __name__ == "__main__":
    source_tokens = "noah offer a drink on the desk on the stool to a child .".split()
    # target_tokens = "<s> * desk ( x _ 6 ) ; * stool ( x _ 9 ) ; drink ( x _ 3 ) and drink . nmod . on ( x _ 3 , x _ 6 ) and desk . nmod . on ( x _ 6 , x _ 9 ) </s>".split()
    # source_tokens = "touch"
    target_tokens = "<s> 111233".split()

    # source_distance_matrix = torch.tensor([[0.0]])
    source_distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 1.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0],
            [2.0, 1.0, 1.0, 0.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0],
            [3.0, 2.0, 4.0, 3.0, 0.0, 2.0, 1.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0],
            [3.0, 2.0, 4.0, 3.0, 2.0, 0.0, 1.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0],
            [2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 0.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0],
            [3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 3.0, 0.0, 2.0, 1.0, 4.0, 4.0, 3.0, 3.0],
            [3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 3.0, 2.0, 0.0, 1.0, 4.0, 4.0, 3.0, 3.0],
            [2.0, 1.0, 3.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0, 3.0, 3.0, 2.0, 2.0],
            [3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 0.0, 2.0, 1.0, 3.0],
            [3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 2.0, 0.0, 1.0, 3.0],
            [2.0, 1.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0, 2.0],
            [2.0, 1.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 0.0],
        ]
    )
    parser = SemanticParser(epsilon=0.25, add_sos_token=False, default_max_distance=7)

    graph, cross_ref = parser.get_dependency_tuples(
        source_tokens=source_tokens, previous_target_tokens=" ".join(target_tokens)
    )
    graph.plot_dependency_tree()
    print(cross_ref)

    for t in cross_ref:
        print(f"------- {t} -----------")
        if len(t) == 2:
            i, j = t[0], t[1]
            tgt_t = target_tokens[i] if i > -1 else -1
            src_t = source_tokens[j] if j > -1 and j < len(source_tokens) else -1
            print(tgt_t, src_t)
        elif len(t) == 3:
            i = t[0]
            tgt_t = target_tokens[i] if i > -1 else -1
            src_t = [
                source_tokens[j] if j > -1 and j < len(source_tokens) else -1
                for j in t[1:]
            ]
            print(tgt_t, src_t)
        else:
            print("meh", t)

    # mask = torch.ones(1, len(target_tokens), dtype=source_distance_matrix.dtype)
    # attn_mask = mask[:, 0]
    # D = source_distance_matrix
    # for i in range(1, len(target_tokens) + 1):
    #     previous_attention_mask = torch.ones_like(mask[:, :i])
    #     previous_attention_mask[:, :-1] = attn_mask
    #     D, attn_mask = parser.get_distance_matrix(
    #         step=i,
    #         previous_cross_distance_matrix=D,
    #         source_distance_matrix=source_distance_matrix.unsqueeze(0),
    #         source_tokens_list=[source_tokens],
    #         previous_target_tokens_list=[target_tokens[:i]],
    #         target_attention_mask_step=previous_attention_mask,
    #     )

    # print(target_tokens[:i])
    # print(source_tokens)
    # print(D)
    # print(attn_mask)

    mask = torch.ones(1, len(target_tokens), dtype=source_distance_matrix.dtype)
    attn_mask_ = mask[:, 0]
    D_ = source_distance_matrix
    for i in range(1, len(target_tokens) + 1):
        print(f"---- {i} -----")
        previous_attention_mask = torch.ones_like(mask[:, :i])
        previous_attention_mask[:, :-1] = attn_mask_
        try:
            D_, attn_mask_ = parser.get_one_distance_matrix(
                step=i,
                previous_cross_distance_matrix=D_,
                source_distance_matrix=source_distance_matrix,
                source_tokens=source_tokens,
                previous_target_tokens=target_tokens[:i],
                target_attention_mask_step=previous_attention_mask[0],
            )
        except Exception as e:
            print(e)
            print(target_tokens[:i])
            print(D_)

    print(D_)
    print(attn_mask_)
    assert torch.allclose(D, D_, rtol=1e-8, atol=1e-8), "not"
