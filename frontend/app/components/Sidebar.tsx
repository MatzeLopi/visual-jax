"use client";

import React, { useEffect, useState } from 'react';
import api from '@/app/lib/api';

// --- Types ---
interface SchemaVariant {
    label: string;
    type: string; // "Dense", "GRU", "Tabular"
    config: any;  // { dim_in: 0, file_path: "" }
}

interface LoadedSchemas {
    inputs: SchemaVariant[];
    layers: SchemaVariant[];
    activations: SchemaVariant[];
}

// --- Schema Parser ---
// Converts Rust's JSON Schema into drag-and-drop items
const parseSchema = (root: any): SchemaVariant[] => {
    // Handle both wrapped schema (root.schema) and direct schema
    const rootDef = root?.schema || root;
    if (!rootDef) return [];

    // 1. Find variants (handling oneOf/anyOf)
    const variants = rootDef.oneOf || rootDef.anyOf || [];

    return variants.map((variant: any) => {
        // 2. Resolve $ref if the variant is just a reference
        let def = variant;
        if (variant.$ref && root.definitions) {
            const refName = variant.$ref.split('/').pop();
            def = root.definitions[refName];
        }

        if (!def || !def.properties) return null;

        // 3. Extract Type Name (e.g., "Dense", "Tabular")
        // Rust's #[serde(tag = "type")] creates: "type": { "const": "Dense" }
        const typeName = def.properties.type?.const;
        if (!typeName) return null;

        // 4. Extract & Build Default Config
        let defaultConfig: any = null;
        if (def.properties.config) {
            defaultConfig = {};

            // Resolve config definition if it's a ref
            let configDef = def.properties.config;
            if (configDef.$ref && root.definitions) {
                const configRefName = configDef.$ref.split('/').pop();
                configDef = root.definitions[configRefName];
            }

            // Generate defaults based on field types
            if (configDef && configDef.properties) {
                Object.entries(configDef.properties).forEach(([key, prop]: [string, any]) => {
                    let propType = prop.type;

                    // Handle nullable/optional types (e.g. ["string", "null"])
                    if (Array.isArray(propType)) {
                        propType = propType.find((t: string) => t !== 'null');
                    }

                    if (propType === 'string') defaultConfig[key] = "";
                    else if (propType === 'integer' || propType === 'number') defaultConfig[key] = 0;
                    else if (propType === 'boolean') defaultConfig[key] = false;
                    else if (propType === 'array') defaultConfig[key] = [];
                    else defaultConfig[key] = null;
                });
            }
        }

        return {
            label: typeName,
            type: typeName,
            config: defaultConfig
        };
    }).filter(Boolean) as SchemaVariant[];
};

export default function Sidebar() {
    const [schemas, setSchemas] = useState<LoadedSchemas>({ inputs: [], layers: [], activations: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch all schemas in parallel
                const [inRes, layerRes, actRes] = await Promise.all([
                    api.get('/compiler/input-types'),
                    api.get('/compiler/layer-types'),
                    api.get('/compiler/activation-types'),
                ]);

                setSchemas({
                    inputs: parseSchema(inRes.data),
                    layers: parseSchema(layerRes.data),
                    activations: parseSchema(actRes.data),
                });
            } catch (err) {
                console.error("Failed to fetch schemas. Is the backend running?", err);
                setError("Failed to load types");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const onDragStart = (event: React.DragEvent, category: string, variant: SchemaVariant) => {
        // Payload: { type: "Dense", config: { dim_in: 0, ... } }
        const configPayload = {
            type: variant.type,
            config: variant.config
        };

        event.dataTransfer.setData('application/reactflow', category); // "Layer", "Input", etc.
        event.dataTransfer.setData('application/config', JSON.stringify(configPayload));
        event.dataTransfer.effectAllowed = 'move';
    };

    // --- Render Item ---
    const DraggableNode = ({ category, variant }: { category: string, variant: SchemaVariant }) => (
        <div
            className="px-3 py-2 mb-2 bg-white border border-gray-200 rounded-md cursor-grab text-xs font-medium text-gray-700 shadow-sm hover:shadow-md hover:border-gray-300 transition-all active:cursor-grabbing flex items-center gap-2 select-none"
            onDragStart={(event) => onDragStart(event, category, variant)}
            draggable
        >
            <div className={`w-2 h-2 rounded-full ${category === 'Input' ? 'bg-purple-500' : category === 'Layer' ? 'bg-gray-900' : 'bg-emerald-500'}`} />
            {variant.label}
        </div>
    );

    if (loading) {
        return (
            <aside className="w-64 bg-gray-50/50 border-r border-gray-200 p-6 h-full flex items-center justify-center">
                <div className="flex flex-col items-center gap-2">
                    <div className="w-4 h-4 border-2 border-gray-300 border-t-black rounded-full animate-spin"></div>
                    <span className="text-xs text-gray-400 font-medium">Loading Backend...</span>
                </div>
            </aside>
        );
    }

    if (error) {
        return (
            <aside className="w-64 bg-gray-50/50 border-r border-gray-200 p-6 h-full flex flex-col items-center justify-center text-center">
                <span className="text-xl mb-2">🔌</span>
                <span className="text-xs text-red-500 font-medium mb-4">{error}</span>
                <button
                    onClick={() => window.location.reload()}
                    className="text-[10px] bg-white border border-gray-300 px-3 py-1 rounded shadow-sm hover:bg-gray-50"
                >
                    Retry
                </button>
            </aside>
        );
    }

    return (
        <aside className="w-64 bg-[#FAFAFA] border-r border-gray-200 p-4 flex flex-col gap-6 h-full overflow-y-auto z-20 custom-scrollbar">

            {/* Inputs */}
            {schemas.inputs.length > 0 && (
                <div>
                    <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 pl-1">Data Source</h3>
                    {schemas.inputs.map(variant => (
                        <DraggableNode key={variant.type} category="Input" variant={variant} />
                    ))}
                </div>
            )}

            {/* Layers */}
            {schemas.layers.length > 0 && (
                <div>
                    <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 pl-1">Layers</h3>
                    {schemas.layers.map(variant => (
                        <DraggableNode key={variant.type} category="Layer" variant={variant} />
                    ))}
                </div>
            )}

            {/* Activations */}
            {schemas.activations.length > 0 && (
                <div>
                    <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 pl-1">Activations</h3>
                    {schemas.activations.map(variant => (
                        <DraggableNode key={variant.type} category="Activation" variant={variant} />
                    ))}
                </div>
            )}
        </aside>
    );
}