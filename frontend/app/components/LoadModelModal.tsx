"use client";

import React, { useState, useEffect } from 'react';
import api from '@/app/lib/api';

export interface Model {
    model_id: string;
    name: string;
    version: number;
    user_id: string | null;
    is_public: boolean;
    created_at: string;
    graph_json: any;
}

interface LoadModelModalProps {
    isOpen: boolean;
    onClose: () => void;
    onLoad: (model: Model) => void;
}

export default function LoadModelModal({ isOpen, onClose, onLoad }: LoadModelModalProps) {
    const [models, setModels] = useState<Model[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [search, setSearch] = useState("");

    useEffect(() => {
        if (isOpen) {
            setLoading(true);
            setError("");
            setSearch("");
            api.get('/models')
                .then(res => setModels(res.data))
                .catch(err => {
                    console.error(err);
                    setError("Failed to load models");
                })
                .finally(() => setLoading(false));
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const filteredModels = models.filter(m => m.name.toLowerCase().includes(search.toLowerCase()));

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-2xl bg-white rounded-lg shadow-xl overflow-hidden border border-gray-200 animate-in fade-in zoom-in duration-200 flex flex-col max-h-[80vh]">
                <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50 shrink-0">
                    <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wide">Load Model</h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="p-4 border-b border-gray-100 shrink-0 relative">
                    <div className="relative">
                        <input
                            type="text"
                            className="w-full px-3 py-2 pl-9 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-black focus:border-black transition-shadow"
                            placeholder="Search models..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            autoFocus
                        />
                        <svg className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                </div>

                <div className="overflow-y-auto flex-1 p-4 bg-gray-50/50 custom-scrollbar">
                    {loading && (
                        <div className="flex flex-col items-center justify-center h-32 text-gray-400 gap-2">
                             <div className="w-5 h-5 border-2 border-gray-300 border-t-black rounded-full animate-spin"></div>
                             <span className="text-xs">Loading models...</span>
                        </div>
                    )}

                    {error && (
                        <div className="text-center text-red-500 py-4 text-sm">{error}</div>
                    )}

                    {!loading && !error && filteredModels.length === 0 && (
                        <div className="text-center text-gray-400 py-8 text-sm">No models found.</div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {filteredModels.map(model => (
                            <div
                                key={model.model_id}
                                onClick={() => { onLoad(model); onClose(); }}
                                className="group bg-white border border-gray-200 rounded-lg p-3 hover:border-black cursor-pointer transition-all hover:shadow-md flex flex-col gap-1 active:scale-[0.98]"
                            >
                                <div className="flex justify-between items-start">
                                    <h4 className="font-bold text-gray-900 text-sm truncate pr-2 group-hover:text-blue-600 transition-colors">{model.name}</h4>
                                    <span className="text-[10px] bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded border border-gray-200 font-mono">v{model.version}</span>
                                </div>
                                <div className="flex justify-between items-center text-[10px] text-gray-500 mt-1">
                                    <span>{new Date(model.created_at).toLocaleDateString()}</span>
                                    <span className={`px-1.5 py-0.5 rounded ${model.is_public ? 'bg-green-50 text-green-700 border border-green-100' : 'bg-gray-100 text-gray-600 border border-gray-200'}`}>
                                        {model.is_public ? 'Public' : 'Private'}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
