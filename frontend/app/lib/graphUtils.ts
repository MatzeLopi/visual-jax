import { Node, Edge } from '@xyflow/react';

export const OUTPUT_KEYS = ['dim_out', 'dim_hidden', 'out_features', 'units', 'output_size'];
export const INPUT_KEYS = ['dim_in', 'in_features', 'input_size'];

interface NodeKind {
    [key: string]: {
        config?: {
            features?: any[];
            [key: string]: any;
        };
        [key: string]: any;
    };
}

export const getOutputDimension = (node: Node, allNodes: Node[], allEdges: Edge[], visited = new Set<string>()): number | null => {
    if (!node || !node.data || !node.data.kind) return null;

    // Prevent infinite loops (cycles)
    if (visited.has(node.id)) return null;
    visited.add(node.id);

    const kind = node.data.kind as NodeKind;
    const kindKey = Object.keys(kind)[0];
    const details = kind[kindKey];

    if (details.config) {
        const outKey = Object.keys(details.config).find(k => OUTPUT_KEYS.includes(k));
        if (outKey) {
            return Number(details.config[outKey]);
        }
    }
    if (kindKey === 'Input' && details.config && Array.isArray(details.config.features)) {
        return details.config.features.length || null;
    }

    // Find incoming edge to trace back
    const incomingEdge = allEdges.find(e => e.target === node.id);
    if (incomingEdge) {
        const parentNode = allNodes.find(n => n.id === incomingEdge.source);
        if (parentNode) {
            return getOutputDimension(parentNode, allNodes, allEdges, visited);
        }
    }

    return null;
};
