"use client";

import React, { useState, useEffect, Suspense, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import api from '../lib/api';
import { Model, Log } from '../types';
import Link from 'next/link';
import { ChevronDown, ChevronRight } from 'lucide-react';

function TrainingContent() {
    const searchParams = useSearchParams();
    const modelIdFromUrl = searchParams.get('modelId');

    const [models, setModels] = useState<Model[]>([]);
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    const [logs, setLogs] = useState<Log[]>([]);
    const [loadingModels, setLoadingModels] = useState(true);
    const [loadingLogs, setLoadingLogs] = useState(false);

    // Store collapsed state for runs. If a run_id is in this set, its logs are hidden.
    const [collapsedRuns, setCollapsedRuns] = useState<Set<string>>(new Set());

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

    const toggleRunCollapse = (runId: string) => {
        setCollapsedRuns(prev => {
            const next = new Set(prev);
            if (next.has(runId)) {
                next.delete(runId);
            } else {
                next.add(runId);
            }
            return next;
        });
    };

    // Group logs by run_id and order them
    const groupedLogs = useMemo(() => {
        if (logs.length === 0) return [];

        // Group by run_id
        const groups: Record<string, Log[]> = {};
        logs.forEach(log => {
            const rid = log.run_id || 'unknown';
            if (!groups[rid]) {
                groups[rid] = [];
            }
            groups[rid].push(log);
        });

        // The API returns logs DESC (newest first).
        // For each group, we want the logs to be chronological (oldest first, i.e., ASC).
        // Since they come in DESC, we reverse them per group.
        for (const rid in groups) {
            groups[rid] = groups[rid].reverse();
        }

        // We want the newest runs first.
        // We can sort the runs by the timestamp of their newest log (which is the last one in the ASC array).
        const sortedRunIds = Object.keys(groups).sort((a, b) => {
            const lastLogA = groups[a][groups[a].length - 1];
            const lastLogB = groups[b][groups[b].length - 1];
            const timeA = lastLogA.created_at ? new Date(lastLogA.created_at).getTime() : 0;
            const timeB = lastLogB.created_at ? new Date(lastLogB.created_at).getTime() : 0;
            return timeB - timeA;
        });

        return sortedRunIds.map(rid => ({
            runId: rid,
            logs: groups[rid]
        }));
    }, [logs]);


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
                                        setCollapsedRuns(new Set());
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

                    <div className="flex-1 overflow-auto bg-[#1E1E1E] text-gray-300 font-mono p-4">
                        {loadingLogs ? (
                            <div className="flex h-full items-center justify-center text-gray-500 italic">
                                Loading logs...
                            </div>
                        ) : logs.length === 0 ? (
                            <div className="flex h-full items-center justify-center text-gray-500 italic">
                                {selectedModel ? "No logs found for this model. Click 'Start Training' to begin." : "Select a model to begin."}
                            </div>
                        ) : (
                            <div className="flex flex-col gap-4">
                                {groupedLogs.map(({ runId, logs: runLogs }, groupIdx) => {
                                    const isCollapsed = collapsedRuns.has(runId);
                                    const firstLogTime = runLogs[0]?.created_at ? new Date(runLogs[0].created_at).toLocaleString() : 'Unknown Time';

                                    return (
                                        <div key={runId} className="border border-gray-700 rounded-md overflow-hidden bg-[#252525]">
                                            <div
                                                className="bg-[#2A2A2A] px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-[#333333] transition-colors"
                                                onClick={() => toggleRunCollapse(runId)}
                                            >
                                                <div className="flex items-center gap-2">
                                                    {isCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                                                    <span className="text-sm font-semibold text-gray-200">
                                                        Run {runId === 'unknown' ? '(Legacy)' : runId.substring(0, 8)}
                                                    </span>
                                                </div>
                                                <div className="text-xs text-gray-400">
                                                    {firstLogTime} • {runLogs.length} events
                                                </div>
                                            </div>

                                            {!isCollapsed && (
                                                <div className="p-3 text-[13px] leading-relaxed whitespace-pre-wrap break-all bg-[#1E1E1E]">
                                                    {runLogs.map((log, idx) => (
                                                        <div key={idx} className="mb-1 hover:bg-[#2A2A2A] px-2 py-0.5 rounded flex items-start">
                                                            <span className="text-gray-500 mr-4 whitespace-nowrap min-w-[80px]">
                                                                [{log.created_at ? new Date(log.created_at).toLocaleTimeString() : '---'}]
                                                            </span>
                                                            <span className={log.text.includes('Error') ? 'text-red-400' : 'text-gray-300'}>
                                                                {log.text}
                                                            </span>
                                                        </div>
                                                    ))}
                                                    {isTraining && groupIdx === 0 && (
                                                        <div className="mt-2 flex items-center text-gray-500 px-2">
                                                            <span className="animate-pulse mr-2">▊</span> Polling for updates...
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
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
