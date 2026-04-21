"use client";

import React, { useState, useCallback, useRef } from 'react';
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
    useReactFlow
} from '@xyflow/react';
import api from '../lib/api';
import Sidebar from '../components/Sidebar';
import PropertiesPanel from '../components/PropertiesPanel';
import TrainingConfig from '../components/TrainingConfig';
import { getOutputDimension, INPUT_KEYS } from '../lib/graphUtils';
import { TrainParams } from '../types';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

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
    const { getNodes, getEdges, screenToFlowPosition } = useReactFlow();
    const router = useRouter();

    // Nodes & Edges State
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [isTrainingModalOpen, setIsTrainingModalOpen] = useState(false);
    const [trainParams, setTrainParams] = useState<TrainParams>({
        loss: { type: '' },
        metrics: [],
        epochs: 10,
        batchsize: 32
    });

    // We don't need to store the instance anymore since we use useReactFlow
    // const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);

    const onConnect = useCallback(
        (params: Connection) => {
            setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#000' } }, eds));

            // Use getNodes and getEdges to access current state without dependency
            const currentNodes = getNodes();
            const currentEdges = getEdges();

            const sourceNode = currentNodes.find((n) => n.id === params.source);
            const targetNode = currentNodes.find((n) => n.id === params.target);

            if (sourceNode && targetNode) {
                const outputDim = getOutputDimension(sourceNode, currentNodes, currentEdges);

                if (outputDim !== null && outputDim > 0) {
                    setNodes((nds) => nds.map((node) => {
                        if (node.id === targetNode.id) {
                            const newKind = JSON.parse(JSON.stringify(node.data.kind));
                            const kindKey = Object.keys(newKind)[0];
                            const details = newKind[kindKey];

                            if (details.config) {
                                // Generic Input Key Search
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
        [getNodes, getEdges, setEdges, setNodes]
    );

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = 'move';
        }
    }, []);

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();
            // if (!reactFlowWrapper.current || !reactFlowInstance) return;
            // We use screenToFlowPosition from hook, checking wrapper ref is good practice but not strictly required for hook if provider is up

            const type = event.dataTransfer?.getData('application/reactflow');
            const configStr = event.dataTransfer?.getData('application/config');

            if (typeof type === 'undefined' || !type) return;

            const configData = JSON.parse(configStr);
            const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });

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
        [screenToFlowPosition, setNodes]
    );

    const onNodeClick = (_: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    };

    // --- Handle Deletion ---
    const handleDeleteNode = (nodeId: string) => {
        setNodes((nds) => nds.filter((n) => n.id !== nodeId));
        setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
        setSelectedNode(null);
    };

    // Also clear selection if user clicks on empty canvas
    const onPaneClick = () => setSelectedNode(null);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleUpdateNode = (nodeId: string, newKind: Record<string, any>) => {
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
            graph: {
                nodes: nodes.map(n => ({ id: n.id, kind: n.data.kind, position: n.position })),
                edges: edges.map(e => ({ id: e.id, source: e.source, target: e.target }))
            },
            params: trainParams
        };

        try {
            const res = await api.post('/compiler/compile', payload);
            setIsTrainingModalOpen(false);
            router.push(`/training?modelId=${res.data.model_id}`);
        } catch (err: unknown) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const msg = (err as any).response?.data?.error || (err as Error).message || "Unknown Error";
            alert(`Error: ${msg}`);
        }
    };

    return (
        <div className="h-screen flex flex-col bg-white text-gray-900 font-sans">
            {/* Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10 shadow-sm">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-3">
                        <div className="w-6 h-6 bg-black rounded flex items-center justify-center">
                            <span className="text-white text-xs font-bold">V</span>
                        </div>
                        <h1 className="font-bold text-sm tracking-tight text-gray-900">
                            Visual JAX <span className="text-gray-400 font-normal">/ Editor</span>
                        </h1>
                    </div>

                    <div className="h-4 w-px bg-gray-300"></div>
                    <Link href="/training" className="text-sm font-medium text-gray-600 hover:text-black transition-colors">
                        Go to Training
                    </Link>
                </div>

                <button
                    onClick={() => setIsTrainingModalOpen(true)}
                    className="bg-black text-white px-5 py-1.5 rounded-md text-xs font-medium hover:bg-gray-800 transition-all shadow-sm"
                >
                    Compile
                </button>
            </div>

            <div className="flex-1 flex overflow-hidden">
                {isTrainingModalOpen && (
                    <TrainingConfig
                        params={trainParams}
                        onChange={setTrainParams}
                        onClose={() => setIsTrainingModalOpen(false)}
                        onStart={handleCompile}
                    />
                )}
                <Sidebar />

                <div className="flex-1 relative bg-[#F9FAFB] h-full shadow-inner" ref={reactFlowWrapper}>
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        // onInit={setReactFlowInstance} // Removed
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
            </div>
        </div>
    );
}
