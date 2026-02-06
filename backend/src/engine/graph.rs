// Validate the Compute Graph before passing it to the engine to compile it in jax/flax
use crate::engine::types::NodeKind;
use crate::schemas::graph::NeuralGraph;
use anyhow::{Result, anyhow};
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

pub struct GraphProcessor {
    pub graph: DiGraph<String, ()>,
    pub node_map: HashMap<String, NodeIndex>,
    pub original_nodes: HashMap<String, NodeKind>,
}

impl GraphProcessor {
    pub fn new(neural_graph: NeuralGraph) -> Self {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();
        let mut original_nodes = HashMap::new();

        // Add Nodes
        for node in neural_graph.nodes {
            let idx = graph.add_node(node.id.clone());
            node_map.insert(node.id.clone(), idx);
            original_nodes.insert(node.id, node.kind);
        }

        // Add Edges
        for edge in neural_graph.edges {
            if let (Some(&src), Some(&target)) =
                (node_map.get(&edge.source), node_map.get(&edge.target))
            {
                graph.add_edge(src, target, ());
            }
        }

        Self {
            graph,
            node_map,
            original_nodes,
        }
    }

    pub fn validate_and_sort(&self) -> Result<Vec<(String, NodeKind)>> {
        // Topological Sort (Checks for cycles automatically)
        let sorted_indices = toposort(&self.graph, None).map_err(|_| {
            anyhow!("Graph contains a cycle! Neural Networks must be acyclic (DAG).")
        })?;

        // Map back to NodeKind
        let sorted_nodes = sorted_indices
            .into_iter()
            .map(|idx| {
                let id = self.graph.node_weight(idx).unwrap().clone();
                let kind = self.original_nodes.get(&id).unwrap().clone();
                (id, kind)
            })
            .collect();

        Ok(sorted_nodes)
    }

    pub fn get_incoming_map(&self) -> HashMap<String, Vec<String>> {
        let mut incoming_map = HashMap::new();

        for (node_id, node_idx) in &self.node_map {
            // Find all incoming edges to this node
            let parents: Vec<String> = self
                .graph
                .neighbors_directed(*node_idx, petgraph::Direction::Incoming)
                .map(|parent_idx| self.graph.node_weight(parent_idx).unwrap().clone())
                .collect();

            incoming_map.insert(node_id.clone(), parents);
        }

        incoming_map
    }
}
