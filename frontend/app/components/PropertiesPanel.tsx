import React from 'react';
import InputProperties, { InputConfig } from './InputProperties';

// Define types for cleaner code
interface NodeData {
    kind: Record<string, any>;
    label: string;
    [key: string]: any;
}

interface SelectedNode {
    id: string;
    data: NodeData;
    [key: string]: any;
}

interface Props {
    selectedNode: SelectedNode | null;
    onChange: (nodeId: string, newKind: Record<string, any>) => void;
    onDelete: (nodeId: string) => void;
}

export default function PropertiesPanel({ selectedNode, onChange, onDelete }: Props) {
    if (!selectedNode) {
        return (
            <div className="w-72 bg-gray-50 border-l border-gray-200 p-4 flex items-center justify-center h-full">
                <div className="text-gray-400 text-sm italic">
                    Select a node to edit properties
                </div>
            </div>
        );
    }

    const kindKey = Object.keys(selectedNode.data.kind)[0];
    const details = selectedNode.data.kind[kindKey];
    const config = details.config || {};

    // --- 1. SPECIAL HANDLER: INPUT NODES ---
    if (kindKey === 'Input') {
        return (
            <InputProperties
                nodeId={selectedNode.id}
                config={config as InputConfig}
                onConfigChange={(newConfig) => {
                    const newKind = {
                        [kindKey]: {
                            ...details,
                            config: newConfig
                        }
                    };
                    onChange(selectedNode.id, newKind);
                }}
                onDelete={() => onDelete(selectedNode.id)}
            />
        );
    }

    // --- 2. GENERIC HANDLER: LAYERS & ACTIVATIONS ---
    const handleGenericChange = (key: string, val: string) => {
        const numericVal = Number(val);
        const finalVal = isNaN(numericVal) || val === '' ? val : numericVal;
        const newConfig = { ...config, [key]: finalVal };
        const newKind = {
            [kindKey]: { ...details, config: newConfig }
        };
        onChange(selectedNode.id, newKind);
    };

    return (
        <div className="w-72 bg-gray-50 border-l border-gray-200 flex flex-col h-full shadow-xl z-20">
            {/* Header */}
            <div className="p-4 border-b bg-white flex justify-between items-center sticky top-0">
                <div>
                    <h3 className="font-bold text-gray-800 text-sm">Properties</h3>
                    <div className="text-[10px] text-gray-400 font-mono">{selectedNode.id}</div>
                </div>
                <button
                    onClick={() => onDelete(selectedNode.id)}
                    className="text-xs text-red-500 hover:text-red-700 font-medium px-2 py-1 hover:bg-red-50 rounded"
                >
                    Delete
                </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1">
                <div className="mb-6">
                    <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">Node Type</label>
                    <div className="text-sm font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded inline-block">
                        {details.type}
                    </div>
                </div>

                {Object.entries(config).map(([key, value]) => (
                    <div key={key} className="mb-4">
                        <label className="block text-xs font-medium text-gray-600 mb-1 capitalize">
                            {key.replace(/_/g, ' ')}
                        </label>
                        <input
                            type="text"
                            defaultValue={value as string}
                            onBlur={(e) => handleGenericChange(key, e.target.value)}
                            className="w-full border border-gray-300 rounded px-3 py-1.5 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
                        />
                    </div>
                ))}

                {Object.keys(config).length === 0 && (
                    <div className="text-xs text-gray-400 italic text-center mt-10">
                        No configurable properties for this node.
                    </div>
                )}
            </div>
        </div>
    );
}
