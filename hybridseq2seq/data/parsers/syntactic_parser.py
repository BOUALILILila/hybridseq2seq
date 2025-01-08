from typing import Optional, List, Dict, Tuple

import stanza

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import numpy as np


class SyntacticParser:
    """Builds a tree structure of words from the input sentence, which represents the syntactic dependency relations between words."""

    def __init__(self) -> None:
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")

    def get_distance_matrix(self, sent: str) -> Tuple[np.ndarray, List[str]]:
        g, vocab, lemmatized_sent = self.get_dependency_tree(sent)
        node_ids = [id for id in vocab]
        dist_matrix = self.get_shortest_dependency_path_lengths(g, node_ids)
        return dist_matrix, lemmatized_sent

    def get_dependency_tree(
        self, sent: str
    ) -> Tuple[nx.Graph, Dict[int, str], List[str]]:
        s_sent = (
            ". " + sent + " ."
        )  # this acts as the sos and eos tokens in the distance matrix but does not change anything else
        # lower cas problem with example: "Emma gave Ava ..." identifies Ava as start of new sentence
        doc = self.nlp(s_sent.lower())  # run annotation over a sentence
        assert len(doc.sentences) == 1, "doc.sentences has more than one sentence"
        ann_sent = doc.sentences[0]
        lemmatized_sent = [
            word.lemma if word.upos == "VERB" else word.text
            for word in ann_sent.words[1:-1]
        ]  # ignore the sos and eos

        g = nx.Graph()
        vocab = {word.id: word.text for word in ann_sent.words}
        # vocab[0] = 'root'
        for word in ann_sent.words:
            g.add_node(word.id)
            if word.head > 0:  # skip root node
                g.add_node(word.head)
                g.add_edge(word.id, word.head)
        return g, vocab, lemmatized_sent

    def get_shortest_dependency_path_lengths(
        self,
        g: nx.Graph,
        node_ids: Optional[List[int]] = None,
        leaves_only: Optional[bool] = False,
    ) -> np.ndarray:
        if node_ids is None:
            node_ids = list(g.nodes())

        if leaves_only:
            print("Considering leaf nodes only.")
            node_ids = [x for x in node_ids if g.degree(x) == 1]  # ordered

        n = len(node_ids)
        # Distance Matrix
        D = np.zeros((n, n))
        for idx, node_id in enumerate(node_ids):
            # Djikstra shortest path ==> lower triangular distance matrix
            p = [
                nx.shortest_path_length(g, source=node_id, target=tgt_id)
                for tgt_id in node_ids[:idx]
            ]
            D[idx, :idx] = np.array(p)
        return D + D.T  # symmetric distance matrix

    def plot_dependency_tree(self, g: nx.graph, vocab: Dict[int, str]) -> None:
        plt.figure(1, figsize=(10, 10))
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, labels=vocab, with_labels=True, node_size=1600, font_size=10)
        plt.show()

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
                in_dist_matrix  # eos is added when constructing the distance matrix
            )
        else:
            exp_in_dist_matrix = in_dist_matrix[:-1,:-1]
        return exp_in_dist_matrix, lem_in_seq_tokens
