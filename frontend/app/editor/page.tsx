"use client";

import { useState, useCallback, useRef, DragEvent } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    ReactFlowProvider,
    Node,
    Edge,
    ReactFlowInstance
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import api from '../lib/api';
import Sidebar from '../components/Sidebar';
import PropertiesPanel from '../components/PropertiesPanel';

let id = 0;
const getId = () => `node_${id++}`;

export default function Editor() {
    return (
        <ReactFlowProvider>
            <EditorContent />
        </ReactFlowProvider>
    );
}

function EditorContent() {
    const reactFlowWrapper = useRef<HTMLDivElement>(null);

    // Nodes & Edges State
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [compiledCode, setCompiledCode] = useState("// Output will appear here...");
    const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);

    // Helper to determine what size a node outputs
    const OUTPUT_KEYS = ['dim_out', 'dim_hidden', 'out_features', 'units', 'output_size'];

    // Now accepts all nodes and edges to perform recursive lookups
    const getOutputDimension = (node: Node, allNodes: Node[], allEdges: Edge[], visited = new Set<string>()): number | null => {
        if (!node || !node.data || !node.data.kind) return null;

        // Prevent infinite loops (cycles)
        if (visited.has(node.id)) return null;
        visited.add(node.id);

        const kind: any = node.data.kind;
        const kindKey = Object.keys(kind)[0];
        const details = kind[kindKey];

        // 1. Explicit Output: Does this node define a size? (e.g., Dense, GRU)
        if (details.config) {
            const outKey = Object.keys(details.config).find(k => OUTPUT_KEYS.includes(k));
            if (outKey) {
                return Number(details.config[outKey]);
            }
        }

        // 2. Tabular Input Special Case
        if (kindKey === 'Input' && details.config && Array.isArray(details.config.features)) {
            return details.config.features.length || null;
        }

        // 3. Pass-Through / Activation: Look backwards!
        // If we are here, this node (e.g. ReLU) has no size config. 
        // We assume it passes through the dimension of its parent.

        // Find the edge connecting TO this node
        const incomingEdge = allEdges.find(e => e.target === node.id);
        if (incomingEdge) {
            const parentNode = allNodes.find(n => n.id === incomingEdge.source);
            if (parentNode) {
                // RECURSION: Ask the parent what IT outputs
                return getOutputDimension(parentNode, allNodes, allEdges, visited);
            }
        }

        return null;
    };

    const onConnect = useCallback(
        (params: Connection) => {
            // 1. Apply the visual edge immediately
            setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#000' } }, eds));

            const sourceNode = nodes.find((n) => n.id === params.source);
            const targetNode = nodes.find((n) => n.id === params.target);

            if (sourceNode && targetNode) {
                // PASS NODES AND EDGES HERE
                // Note: We use 'edges' from state, which doesn't include the NEW edge yet if we act immediately.
                // However, for "Source -> Target", we only care about what feeds INTO Source.
                // If Source is ReLU, it must already be connected to something for this to work.
                const outputDim = getOutputDimension(sourceNode, nodes, edges);

                if (outputDim !== null && outputDim > 0) {
                    setNodes((nds) => nds.map((node) => {
                        if (node.id === targetNode.id) {
                            const newKind = JSON.parse(JSON.stringify(node.data.kind));
                            const kindKey = Object.keys(newKind)[0];
                            const details = newKind[kindKey];

                            if (details.config) {
                                // Generic Input Key Search
                                const INPUT_KEYS = ['dim_in', 'in_features', 'input_size'];
                                const inKey = Object.keys(details.config).find(k => INPUT_KEYS.includes(k));

                                if (inKey) {
                                    details.config[inKey] = outputDim;
                                    return { ...node, data: { ...node.data, kind: newKind } };
                                }
                            }
                        }
                        return node;
                    }));
                }
            }
        },
        [nodes, edges, setEdges, setNodes] // Added 'edges' to dependencies
    );

    const onDragOver = useCallback((event: DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event: DragEvent) => {
            event.preventDefault();
            if (!reactFlowWrapper.current || !reactFlowInstance) return;

            const type = event.dataTransfer.getData('application/reactflow');
            const configStr = event.dataTransfer.getData('application/config');

            if (typeof type === 'undefined' || !type) return;

            const configData = JSON.parse(configStr);
            const position = reactFlowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY });

            const kindPayload = { [type]: configData };

            const newNode: Node = {
                id: getId(),
                type: type === 'Input' ? 'input' : 'default',
                position,
                data: { label: `${configData.type}`, kind: kindPayload },
                style: {
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    background: '#fff',
                    padding: '10px',
                    minWidth: '100px',
                    boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1)'
                }
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance, setNodes]
    );

    const onNodeClick = (_: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    };

    // --- NEW: Handle Deletion ---
    const handleDeleteNode = (nodeId: string) => {
        setNodes((nds) => nds.filter((n) => n.id !== nodeId));
        setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
        setSelectedNode(null);
    };

    // Also clear selection if user clicks on empty canvas
    const onPaneClick = () => setSelectedNode(null);

    const handleUpdateNode = (nodeId: string, newKind: any) => {
        setNodes((nds) => nds.map((node) => {
            if (node.id === nodeId) {
                const innerKey = Object.keys(newKind)[0];
                const innerData = newKind[innerKey];
                return {
                    ...node,
                    data: { ...node.data, kind: newKind, label: innerData.type }
                };
            }
            return node;
        }));
        setSelectedNode((prev) => prev && prev.id === nodeId ? { ...prev, data: { ...prev.data, kind: newKind } } : prev);
    };

    const handleCompile = async () => {
        const payload = {
            nodes: nodes.map(n => ({ id: n.id, kind: n.data.kind, position: n.position })),
            edges: edges.map(e => ({ id: e.id, source: e.source, target: e.target }))
        };

        try {
            const res = await api.post('/compiler/compile', payload);
            setCompiledCode(res.data.code);
        } catch (err: any) {
            const msg = err.response?.data?.error || err.message || "Unknown Error";
            setCompiledCode(`Error: ${msg}`);
        }
    };

    return (
        <div className="h-screen flex flex-col bg-white text-gray-900 font-sans">
            {/* Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 bg-black rounded flex items-center justify-center">
                        <span className="text-white text-xs font-bold">V</span>
                    </div>
                    <h1 className="font-bold text-sm tracking-tight text-gray-900">Visual JAX <span className="text-gray-400 font-normal">/ Editor</span></h1>
                </div>
                <button
                    onClick={handleCompile}
                    className="bg-black text-white px-5 py-1.5 rounded-md text-xs font-medium hover:bg-gray-800 transition-all shadow-sm"
                >
                    Compile Graph
                </button>
            </div>

            <div className="flex-1 flex overflow-hidden">
                <Sidebar />

                <div className="flex-1 relative bg-[#F9FAFB] h-full shadow-inner" ref={reactFlowWrapper}>
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        onInit={setReactFlowInstance}
                        onDrop={onDrop}
                        onDragOver={onDragOver}
                        onNodeClick={onNodeClick}
                        onPaneClick={onPaneClick}
                        deleteKeyCode={["Backspace", "Delete"]}
                        fitView
                    >
                        <Background gap={24} size={1} color="#E5E7EB" />
                        <Controls className="bg-white border-gray-200 shadow-sm rounded-lg text-gray-600" />
                    </ReactFlow>
                </div>

                <PropertiesPanel
                    selectedNode={selectedNode}
                    onChange={handleUpdateNode}
                    onDelete={handleDeleteNode}
                />

                <div className="w-[400px] border-l border-gray-200 bg-white flex flex-col z-20">
                    <div className="px-4 py-3 bg-white text-gray-900 text-xs font-bold border-b border-gray-100 flex justify-between items-center tracking-wide">
                        GENERATED PYTHON
                    </div>
                    <div className="flex-1 overflow-auto p-0 bg-[#FAFAFA]">
                        <pre className="text-[11px] font-mono text-gray-600 p-4 leading-relaxed whitespace-pre overflow-x-auto">
                            {compiledCode}
                        </pre>
                    </div>
                </div>
            </div>
        </div>
    );
}