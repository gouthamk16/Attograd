"""
Computation graph visualization tools.
"""

from graphviz import Digraph, ExecutableNotFound
from ..tensor import Tensor


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, filename='expression_graph'):
    """
    Render the computation graph rooted at `root` and save to `filename`.
    Requires the graphviz system binaries to be installed.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        data_str = f'{float(n.data):.4f}' if n.data.ndim == 0 else str(n.data.shape)
        grad_str = f'{float(n.grad):.4f}' if hasattr(n.grad, '__float__') else str(n.grad)
        dot.node(name=uid, label=f'{{ {n.label} | data {data_str} | grad {grad_str} }}', shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    try:
        out_path = dot.render(filename, cleanup=True)
        print(f"Graph saved to {out_path}")
    except ExecutableNotFound:
        raise RuntimeError(
            "Graphviz system binaries not found. "
            "Install from https://graphviz.org/download/ and ensure 'dot' is on PATH."
        )
