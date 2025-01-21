from typing import Optional, List, Dict, Tuple
import numpy as np

from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from lark import Discard, Lark, Token, Transformer, v_args


class DependencyGraph:
    def __init__(self):
        self._graph = nx.Graph()
        self._words = {}
        self.is_primitive = False

    def get_words(self):
        return self._words

    def get_words_list(self):
        l = [self._words[i] for i in range(len(self._words))]  # words in order
        assert len(l) == len(self._graph.nodes())
        return l

    def add_edge(self, token_1: int, token_2: int):
        self._graph.add_edge(token_1, token_2)

    def add_node(self, token: str):
        idx = len(self._graph)
        self._graph.add_node(idx)
        self._words[idx] = token
        return idx

    def plot(self):
        plt.figure(1, figsize=(5, 5))
        g = self._graph
        pos = graphviz_layout(g, prog="dot")
        nx.draw(
            g, pos, with_labels=True, labels=self._words, node_size=1000, font_size=10
        )
        labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.show()

    def get_distance_matrix(self):
        # Distance Matrix
        n = len(self._graph)
        D = np.zeros((n, n))
        for idx in range(n):
            # Djikstra shortest path ==> lower triangular distance matrix
            p = [
                nx.shortest_path_length(
                    self._graph,
                    source=idx,
                    target=tgt_idx,
                )
                for tgt_idx in range(idx)
            ]
            D[idx, :idx] = np.array(p)
        D_with_sos = np.zeros(
            (n + 1, n + 1)
        )  # for the sos token which is not in the graph (has the same distances as the eos)
        D_with_sos[1:, 1:] = D + D.T  # symmetric distance matrix
        D_with_sos[0, 1:n] = D_with_sos[
            n, 1:n
        ]  # Copy end point distances to sos distances
        D_with_sos[0, n] = (
            2  # distance from end point to sos token (both related to the root => d = 2)
        )
        D_with_sos[1 : n + 1, 0] = D_with_sos[0, 1 : n + 1]
        return D_with_sos


@v_args(inline=True)  # Affects the signatures of the methods
class PrecolTreeTransformer(Transformer):
    """Visits the abstract syntax tree of the source sequence and extracts the constituency
    tree.
    """

    def __init__(self):
        self.dep_graph = DependencyGraph()
        self.words = self.dep_graph.get_words()
        self.lemmatizer = WordNetLemmatizer()  # Lematize the verbs

    def s_tree(self, root_node: int) -> int:
        return root_node

    def primitive(self, root_node: int) -> int:
        self.dep_graph.is_primitive = True
        return root_node

    # S percol
    def head_vp(self, vp: int) -> int:
        return vp

    def np_vp_head_vp(self, np_: int, vp: int) -> int:
        self.dep_graph.add_edge(np_, vp)
        return vp

    # NP percol
    def head_n(self, node: int) -> int:
        return node

    def det_n_pp_head_n(self, det: int, n: int, pp: int) -> int:
        self.dep_graph.add_edge(det, n)
        self.dep_graph.add_edge(pp, n)
        return n

    def det_n_head_n(self, det: int, n: int) -> int:
        self.dep_graph.add_edge(det, n)
        return n

    # VP percol
    def head_v(self, node: int) -> int:
        return node

    def v_np_head_v(self, v: int, np_: int) -> int:
        self.dep_graph.add_edge(np_, v)
        return v

    def v_np_np_head_v(self, v: int, np_1: int, np_2: int) -> int:
        self.dep_graph.add_edge(np_1, v)
        self.dep_graph.add_edge(np_2, v)
        return v

    def v_np_pp_head_v(self, v: int, np_: int, pp: int) -> int:
        self.dep_graph.add_edge(np_, v)
        self.dep_graph.add_edge(pp, v)
        return v

    def v_c_s_head_v(self, v: int, c: int, s: int) -> int:
        self.dep_graph.add_edge(c, v)
        self.dep_graph.add_edge(s, v)
        return v

    def v_inf_v_head_v(self, v: int, inf: int, v_: int) -> int:
        self.dep_graph.add_edge(v_, v)
        self.dep_graph.add_edge(inf, v)
        return v

    def np_v_head_v(self, np_: int, v: int) -> int:
        self.dep_graph.add_edge(np_, v)
        return v

    def aux_v_by_np_head_v(self, aux: int, v: int, by: int, np_: int) -> int:
        self.dep_graph.add_edge(aux, v)
        self.dep_graph.add_edge(by, v)
        self.dep_graph.add_edge(np_, v)
        return v

    def aux_v_np_head_v(self, aux: int, v: int, np_: int) -> int:
        self.dep_graph.add_edge(aux, v)
        self.dep_graph.add_edge(np_, v)
        return v

    def aux_v_pp_head_v(self, aux: int, v: int, pp: int) -> int:
        self.dep_graph.add_edge(aux, v)
        self.dep_graph.add_edge(pp, v)
        return v

    def aux_v_np_by_np_head_v(
        self, aux: int, v: int, np_1: int, by: int, np_2: int
    ) -> int:
        self.dep_graph.add_edge(aux, v)
        self.dep_graph.add_edge(np_1, v)
        self.dep_graph.add_edge(by, v)
        self.dep_graph.add_edge(np_2, v)
        return v

    def aux_v_pp_by_np_head_v(
        self, aux: int, v: int, pp: int, by: int, np_: int
    ) -> int:
        self.dep_graph.add_edge(aux, v)
        self.dep_graph.add_edge(pp, v)
        self.dep_graph.add_edge(by, v)
        self.dep_graph.add_edge(np_, v)
        return v

    def aux_v_head_v(self, aux: int, v: int) -> int:
        self.dep_graph.add_edge(aux, v)
        return v

    # PP percol
    def p_np_head_np(self, p: int, np_: int):
        self.dep_graph.add_edge(p, np_)
        return np_

    # Nodes
    def s(self, root: int) -> int:
        return root

    def pp(self, node: int) -> int:
        return node

    def vp(self, node: int) -> int:
        return node

    def np(self, node: int) -> int:
        return node

    def v(self, node: int) -> int:
        verb = self.words[node]
        self.words[node] = self.lemmatizer.lemmatize(verb, "v")
        return node

    def n(self, node: int) -> int:
        return node

    def p(self, node: int) -> int:
        return node

    def c(self, node: int) -> int:
        return node

    def det(self, node: int) -> int:
        return node

    def inf(self, node: int) -> int:
        return node

    def aux(self, node: int) -> int:
        return node

    def by(self, node: int) -> int:
        return node

    def WORD(self, token: Token) -> int:
        word = token.value.strip()
        idx = self.dep_graph.add_node(word)
        return idx

    def __default__(self, data, children, meta):
        return children

    def WS(self, token: Token):
        return Discard

    def __default_token__(self, token: Token):
        return Discard


class GoldSyntaxParser:
    """Builds a tree structure of words from the Gold syntax tree of the input sentence from Syntax-COGS."""

    def __init__(self) -> None:
        ast_grammar = r"""
                s_tree      : s                             
                            | primitive                    
                
                primitive   : v | n 

                s           : "(" WS "S" WS ins WS ")"
                ins         : vp                            ->  head_vp
                            | np WS vp                      ->  np_vp_head_vp

                np          : "(" WS "NP" WS innp WS ")"
                innp        : det WS n WS pp                -> det_n_pp_head_n
                            | det WS n                      -> det_n_head_n
                            | n                             -> head_n

                vp          : "(" WS "VP" WS invp WS ")"
                invp        : v WS np WS np                 -> v_np_np_head_v
                            | v WS np WS pp                 -> v_np_pp_head_v
                            | v WS np                       -> v_np_head_v
                            | v                             -> head_v
                            | v WS c WS s                   -> v_c_s_head_v
                            | v WS inf WS v                 -> v_inf_v_head_v
                            | np WS v                       -> np_v_head_v
                            | aux WS v WS by WS np          -> aux_v_by_np_head_v
                            | aux WS v WS np                -> aux_v_np_head_v
                            | aux WS v WS pp                -> aux_v_pp_head_v
                            | aux WS v WS np WS by WS np    -> aux_v_np_by_np_head_v
                            | aux WS v WS pp WS by WS np    -> aux_v_pp_by_np_head_v
                            | aux WS v                      -> aux_v_head_v

                pp          : "(" WS "PP" WS inpp WS ")"
                inpp        : p WS np                       -> p_np_head_np

                v           : "(" WS "V" WS WORD WS ")"
                n           : "(" WS "N" WS WORD WS ")"
                p           : "(" WS "P" WS WORD WS ")"
                det         : "(" WS "Det" WS WORD WS ")"
                c           : "(" WS "C" WS WORD WS ")"
                inf         : "(" WS "INF" WS WORD WS ")"
                aux         : "(" WS "AUX" WS WORD WS ")"
                by          : "(" WS "BY" WS WORD WS ")"

                WORD      : /[A-Za-z]+/
                WS        : " "
        """
        self.ast_parser = Lark(ast_grammar, start="s_tree")

    def get_distance_matrix(
        self, inline_syntax_tree: str
    ) -> Tuple[np.ndarray, List[str]]:
        dep_graph, lemmatized_sent = self.get_dependency_tree(inline_syntax_tree)
        return dep_graph.get_distance_matrix(), lemmatized_sent

    def get_dependency_tree(
        self, inline_syntax_tree: str
    ) -> Tuple[DependencyGraph, List[str]]:
        ast = self.ast_parser.parse(inline_syntax_tree)
        transformer = PrecolTreeTransformer()
        root_node = transformer.transform(ast)
        dep_graph = transformer.dep_graph
        # Add the end point nodes and link it to the root node
        if not dep_graph.is_primitive:
            idx_end_point = dep_graph.add_node(".")
            dep_graph.add_edge(idx_end_point, root_node)
        lemmatized_sent = dep_graph.get_words_list()
        # Add end of sequence node
        idx_eos = dep_graph.add_node("</s>")
        dep_graph.add_edge(idx_eos, root_node)
        return dep_graph, lemmatized_sent

    def manage_sos_token(
        self,
        in_dist_matrix: np.ndarray,
        lem_in_seq_tokens: List[str],
        add_sos_token: bool,
        sos_token: str,
    ) -> Tuple[np.ndarray, List[str]]:
        if add_sos_token:
            lem_in_seq_tokens = [sos_token] + lem_in_seq_tokens
            exp_in_dist_matrix = (
                in_dist_matrix  # sos is added when constructing the distance matrix
            )
        else:
            exp_in_dist_matrix = in_dist_matrix[1:, 1:]  # remove start token distances
        return exp_in_dist_matrix, lem_in_seq_tokens

    def manage_eos_token(
        self,
        in_dist_matrix: np.ndarray,
        lem_in_seq_tokens: List[str],
        add_eos_token: bool,
        eos_token: str,
    ) -> Tuple[np.ndarray, List[str]]:
        if add_eos_token:
            lem_in_seq_tokens += [eos_token]
            exp_in_dist_matrix = (
                in_dist_matrix  # eos is added when constructing the dependency graph
            )
        else:
            exp_in_dist_matrix = in_dist_matrix[:-1, :-1]
        return exp_in_dist_matrix, lem_in_seq_tokens
