"""Small topology tests; run with python -m unittest tools.test_hoct_graph_fusion."""
import unittest
from tools.hoct_graph_fusion import fuse_hoct_forks


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {1: {"t": 0}, 2: {"t": 1}, 3: {"t": 1}, 4: {"t": 0}, 5: {"t": 1}}
        self.edges = [dict(source_id=1, target_id=2)]
        self.pairs = {(1, 2), (1, 3)}

    def test_orphan(self):
        edges, edits = fuse_hoct_forks(self.nodes, self.edges, self.pairs)
        self.assertEqual(len(edges), 2)
        self.assertEqual(len(edits), 1)

    def test_ordinary_reparent_requires_opt_in(self):
        edges = self.edges + [dict(source_id=4, target_id=3)]
        self.assertFalse(fuse_hoct_forks(self.nodes, edges, self.pairs)[1])
        result, edits = fuse_hoct_forks(self.nodes, edges, self.pairs, True)
        self.assertEqual({(e["source_id"], e["target_id"]) for e in result}, self.pairs)
        self.assertEqual(edits[0]["previous_source_id"], 4)

    def test_existing_division_is_protected(self):
        edges = self.edges + [dict(source_id=4, target_id=3), dict(source_id=4, target_id=5)]
        self.assertFalse(fuse_hoct_forks(self.nodes, edges, self.pairs, True)[1])

    def test_empty_proposals(self):
        self.assertFalse(fuse_hoct_forks(self.nodes, self.edges, set())[1])

    def test_invalid_time_is_rejected(self):
        with self.assertRaises(AssertionError):
            fuse_hoct_forks(self.nodes, self.edges, {(2, 3)})


if __name__ == "__main__":
    unittest.main()
