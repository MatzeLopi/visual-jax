"use client";

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import api from '../lib/api';
import { Model, Log } from '../types';
import Link from 'next/link';

function TrainingContent() {
    const searchParams = useSearchParams();
    const modelIdFromUrl = searchParams.get('modelId');

    const [models, setModels] = useState<Model[]>([]);
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    const [logs, setLogs] = useState<Log[]>([]);
    const [loadingModels, setLoadingModels] = useState(true);
    const [loadingLogs, setLoadingLogs] = useState(false);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const res = await api.get('/models', { data: {} });
                setModels(res.data);

                if (modelIdFromUrl) {
                    const found = res.data.find((m: Model) => m.model_id === modelIdFromUrl);
                    if (found) {
                        setSelectedModel(found);
                    }
                }
            } catch (err) {
                console.error("Error fetching models:", err);
            } finally {
                setLoadingModels(false);
            }
        };

        fetchModels();
    }, [modelIdFromUrl]);

    useEffect(() => {
        const fetchHistoricLogs = async () => {
            if (selectedModel && !isTraining) {
                setLoadingLogs(true);
                try {
                    const res = await api.get(`/logs/${selectedModel.model_id}`);
                    setLogs(res.data);
                } catch (err) {
                    console.error("Error fetching historic logs:", err);
                } finally {
                    setLoadingLogs(false);
                }
            }
        };
        fetchHistoricLogs();
    }, [selectedModel, isTraining]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isTraining && selectedModel) {
            interval = setInterval(async () => {
                try {
                    const res = await api.get(`/logs/${selectedModel.model_id}`);
                    setLogs(res.data);

                    if (res.data.some((l: Log) => l.text.includes("Container stopped."))) {
                        setIsTraining(false);
                    }
                } catch (err) {
                    console.error("Error fetching logs:", err);
                }
            }, 2000);
        }
        return () => clearInterval(interval);
    }, [isTraining, selectedModel]);

    const handleStartTraining = async () => {
        if (!selectedModel) return;
        try {
            await api.post('/training/start', selectedModel);
            setIsTraining(true);
            setLogs([]); // Clear previous logs
        } catch (err: unknown) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const msg = (err as any).response?.data?.error || (err as Error).message || "Unknown Error";
            alert(`Error starting training: ${msg}`);
        }
    };

    return (
        <div className="h-screen flex flex-col bg-[#F9FAFB] text-gray-900 font-sans">
            {/* Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10 shadow-sm">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-3">
                        <div className="w-6 h-6 bg-black rounded flex items-center justify-center">
                            <span className="text-white text-xs font-bold">V</span>
                        </div>
                        <h1 className="font-bold text-sm tracking-tight text-gray-900">
                            Visual JAX <span className="text-gray-400 font-normal">/ Training</span>
                        </h1>
                    </div>

                    <div className="h-4 w-px bg-gray-300"></div>
                    <Link href="/dashboard" className="text-sm font-medium text-gray-600 hover:text-black transition-colors">
                        Dashboard
                    </Link>
                    <Link href="/editor" className="text-sm font-medium text-gray-600 hover:text-black transition-colors">
                        Editor
                    </Link>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden p-6 gap-6">
                {/* Left Sidebar for Model Selection */}
                <div className="w-80 bg-white border border-gray-200 rounded-lg shadow-sm flex flex-col overflow-hidden">
                    <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                        <h2 className="text-sm font-bold text-gray-700">Available Models</h2>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2">
                        {loadingModels ? (
                            <div className="text-center text-sm text-gray-500 py-4">Loading models...</div>
                        ) : models.length === 0 ? (
                            <div className="text-center text-sm text-gray-500 py-4">No models found.</div>
                        ) : (
                            models.map((model) => (
                                <div
                                    key={`${model.model_id}-${model.version_}`}
                                    onClick={() => {
                                        setSelectedModel(model);
                                        setLogs([]);
                                        setIsTraining(false);
                                    }}
                                    className={`p-3 mb-2 rounded-md cursor-pointer border transition-colors ${
                                        selectedModel?.model_id === model.model_id
                                            ? 'border-blue-500 bg-blue-50'
                                            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                                    }`}
                                >
                                    <div className="text-sm font-semibold truncate">
                                        {model.model_name || `Model ${model.model_id.substring(0, 8)}`}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        Version: {model.version_}
                                    </div>
                                    <div className="text-[10px] text-gray-400 mt-1 font-mono truncate">
                                        ID: {model.model_id}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Right Area for Training Monitor */}
                <div className="flex-1 bg-white border border-gray-200 rounded-lg shadow-sm flex flex-col overflow-hidden">
                    <div className="px-6 py-4 bg-gray-50 border-b border-gray-200 flex justify-between items-center">
                        <h2 className="text-sm font-bold text-gray-700">Training Monitor</h2>
                        <button
                            onClick={handleStartTraining}
                            disabled={!selectedModel || isTraining}
                            className={`px-6 py-2 rounded-md text-sm font-medium transition-all ${
                                !selectedModel || isTraining
                                    ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                                    : 'bg-green-600 text-white hover:bg-green-700 shadow-sm'
                            }`}
                        >
                            {isTraining ? 'Training...' : selectedModel ? 'Start Training' : 'Select a Model'}
                        </button>
                    </div>

                    <div className="flex-1 overflow-auto bg-[#1E1E1E] text-gray-300 font-mono p-6">
                        {loadingLogs ? (
                            <div className="flex h-full items-center justify-center text-gray-500 italic">
                                Loading logs...
                            </div>
                        ) : logs.length === 0 ? (
                            <div className="flex h-full items-center justify-center text-gray-500 italic">
                                {selectedModel ? "No logs found for this model. Click 'Start Training' to begin." : "Select a model to begin."}
                            </div>
                        ) : (
                            <div className="text-[13px] leading-relaxed whitespace-pre-wrap break-all">
                                {logs.map((log, idx) => (
                                    <div key={idx} className="mb-1 hover:bg-[#2A2A2A] px-2 py-0.5 rounded">
                                        <span className="text-gray-500 mr-4">[{log.created_at ? new Date(log.created_at).toLocaleTimeString() : '---'}]</span>
                                        <span className={log.text.includes('Error') ? 'text-red-400' : 'text-gray-300'}>{log.text}</span>
                                    </div>
                                ))}
                                {isTraining && (
                                    <div className="mt-4 flex items-center text-gray-500 px-2">
                                        <span className="animate-pulse mr-2">▊</span> Polling for updates...
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function TrainingPage() {
    return (
        <Suspense fallback={<div className="h-screen flex items-center justify-center bg-gray-50">Loading...</div>}>
            <TrainingContent />
        </Suspense>
    );
}
