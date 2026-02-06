import React from 'react';

export default function PropertiesPanel({ selectedNode, onChange }: any) {
    if (!selectedNode) {
        return (
            <div className="p-4 text-gray-400 text-sm text-center italic">
                Select a node to edit properties
            </div>
        );
    }

    // Extract the specific config object based on node structure
    // Structure: node.data.kind = { Layer: { type: 'Dense', config: { ... } } }
    const kindKey = Object.keys(selectedNode.data.kind)[0]; // "Layer", "Input", or "Activation"
    const details = selectedNode.data.kind[kindKey];

    // If it's a simple enum (like Relu), details might just be the string "Relu" or an object
    // Based on your payload, Activation is usually { "Relu": null } or similar, 
    // but let's assume the Drag payload structure: { type: 'Dense', config: {...} }

    // Helper to update nested config
    const handleConfigChange = (key: string, val: any) => {
        const newConfig = { ...details.config, [key]: isNaN(Number(val)) ? val : Number(val) };

        // Reconstruct the full kind object
        const newKind = {
            [kindKey]: {
                ...details,
                config: newConfig
            }
        };

        onChange(selectedNode.id, newKind);
    };

    return (
        <div className="w-72 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
            <h3 className="font-bold text-gray-800 mb-4 border-b pb-2">Properties</h3>

            <div className="mb-4">
                <label className="block text-xs font-bold text-gray-500 uppercase">ID</label>
                <div className="text-sm font-mono text-gray-700">{selectedNode.id}</div>
            </div>

            <div className="mb-4">
                <label className="block text-xs font-bold text-gray-500 uppercase">Type</label>
                <div className="text-sm font-medium text-blue-600">{details.type}</div>
            </div>

            {/* Render Inputs dynamically based on config */}
            {details.config && Object.entries(details.config).map(([key, value]) => (
                <div key={key} className="mb-3">
                    <label className="block text-xs text-gray-600 mb-1 capitalize">{key.replace('_', ' ')}</label>
                    <input
                        type="text"
                        defaultValue={value as string}
                        onBlur={(e) => handleConfigChange(key, e.target.value)}
                        className="w-full border border-gray-300 rounded px-2 py-1 text-sm focus:border-blue-500 outline-none"
                    />
                </div>
            ))}

            {!details.config && (
                <div className="text-xs text-gray-400">No configurable properties.</div>
            )}
        </div>
    );
}