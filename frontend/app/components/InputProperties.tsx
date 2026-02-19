import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import api from '../lib/api';

interface Dataset {
    dataset_id: string;
    name: string;
    file_path: string;
    is_public: boolean;
    created_at?: string;
}

export interface InputConfig {
    features?: string[];
    targets?: string[];
    target?: string;
    sequence_length?: number;
    file_path?: string;
    dim_out?: number;
    [key: string]: any;
}

interface InputProps {
    nodeId: string;
    config: InputConfig;
    onConfigChange: (newConfig: InputConfig) => void;
    onDelete: () => void;
}

export default function InputProperties({ nodeId, config, onConfigChange, onDelete }: InputProps) {
    const [headers, setHeaders] = useState<string[]>([]);
    const [availableDatasets, setAvailableDatasets] = useState<Dataset[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [isLoadingContent, setIsLoadingContent] = useState(false);

    // Config State
    const features: string[] = config.features || [];
    const targets: string[] = Array.isArray(config.targets)
        ? config.targets
        : (config.target ? [config.target] : []);
    const seqLen: number = config.sequence_length || 1;

    // 1. Fetch Datasets on Mount
    useEffect(() => {
        api.get('/datasets/list')
            .then(res => setAvailableDatasets(res.data))
            .catch(err => console.error("Failed to fetch datasets", err));
    }, []);

    // 2. Helper: Client-Side Header Parsing (Preview)
    // Accepts a File (or Blob) and extracts headers
    const parseHeadersLocally = (file: File | Blob, fileName: string) => {
        setHeaders([]); // Clear previous headers

        if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                const data = evt.target?.result;
                if (!data) return;
                try {
                    const workbook = XLSX.read(data, { type: 'array' });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];
                    const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
                    if (json && json.length > 0) setHeaders(json[0] as string[]);
                } catch (err) {
                    console.error("Excel parse error:", err);
                }
            };
            reader.readAsArrayBuffer(file);
        } else {
            // Treat as CSV by default
            // PapaParse accepts File or string. We pass the File object.
            Papa.parse(file as File, {
                header: true,
                preview: 1,
                complete: (results) => {
                    if (results.meta.fields) setHeaders(results.meta.fields);
                },
                error: (err) => console.error("CSV Parse Error:", err)
            });
        }
    };

    // 3. Handle NEW File Upload
    const handleNewUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Preview headers immediately from local file
        parseHeadersLocally(file, file.name);

        setIsUploading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);

            // POST to /datasets (creates DB entry + saves file)
            const res = await api.post('/datasets/upload', formData);
            const newDataset: Dataset = res.data;

            // Add to list and select it
            setAvailableDatasets(prev => [newDataset, ...prev]);

            // Link config to the new server path
            onConfigChange({
                ...config,
                file_path: newDataset.file_path,
                features: [],
                targets: [],
                target: undefined, // Clear legacy
                sequence_length: 1
            });

        } catch (error) {
            console.error("Upload failed", error);
            alert("Failed to upload file. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    // 4. Handle SELECTING Existing Dataset
    const handleSelectDataset = async (path: string) => {
        const ds = availableDatasets.find(d => d.file_path === path);
        if (!ds) return;

        // Update UI immediately to show selected file
        onConfigChange({
            ...config,
            file_path: ds.file_path,
            features: [],
            targets: [],
            target: undefined,
            sequence_length: 1
        });
        setHeaders([]); // Clear headers until fetched

        setIsLoadingContent(true);
        try {
            // Fetch the file content from backend to parse headers
            const res = await api.get(`/datasets/${ds.dataset_id}/file`, {
                responseType: 'blob' // Important: Get raw binary data
            });

            // Create a File object from the blob so our parser logic can reuse it
            const file = new File([res.data], ds.name);

            // Parse headers
            parseHeadersLocally(file, ds.name);

        } catch (err) {
            console.error("Failed to load dataset content", err);
            // Don't alert if it's just a permissions check failing silently,
            // but usually we want user to know why columns didn't appear.
            alert("Could not load dataset columns. You might not have permission to view this file's content.");
        } finally {
            setIsLoadingContent(false);
        }
    };

    const toggleFeature = (col: string) => {
        let newFeatures = [...features];
        if (newFeatures.includes(col)) {
            newFeatures = newFeatures.filter(f => f !== col);
        } else {
            newFeatures.push(col);
        }
        onConfigChange({
            ...config,
            features: newFeatures,
            dim_out: newFeatures.length
        });
    };

    const toggleTarget = (col: string) => {
        let newTargets = [...targets];
        if (newTargets.includes(col)) {
            newTargets = newTargets.filter(t => t !== col);
        } else {
            newTargets.push(col);
        }
        onConfigChange({
            ...config,
            targets: newTargets,
            target: undefined
        });
    };

    return (
        <div className="w-72 bg-white border-l border-gray-200 flex flex-col h-full shadow-xl">
            {/* Header */}
            <div className="p-4 border-b bg-gray-50 flex justify-between items-center">
                <div>
                    <h3 className="font-bold text-gray-800 text-sm">Data Input</h3>
                    <div className="text-[10px] text-gray-400 font-mono">{nodeId}</div>
                </div>
                <button onClick={onDelete} className="text-xs text-red-500 font-medium hover:text-red-700">
                    Delete
                </button>
            </div>

            <div className="p-4 overflow-y-auto space-y-6 flex-1">

                {/* 1. Dataset Selection */}
                <div>
                    <label className="block text-[10px] font-bold text-gray-500 uppercase mb-2">
                        Select Dataset
                    </label>
                    <select
                        className="w-full text-xs border border-gray-300 rounded p-2 mb-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                        onChange={(e) => handleSelectDataset(e.target.value)}
                        value={config.file_path || ""}
                        disabled={isLoadingContent}
                    >
                        <option value="" disabled>-- Choose Data --</option>
                        {availableDatasets.map(ds => (
                            <option key={ds.dataset_id} value={ds.file_path}>
                                {ds.name} {ds.is_public ? "(Public)" : ""}
                            </option>
                        ))}
                    </select>

                    {/* New Upload Button masking a file input */}
                    <div className="relative group">
                        <input
                            type="file"
                            accept=".csv,.xlsx,.xls"
                            onChange={handleNewUpload}
                            disabled={isUploading}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                        />
                        <button className="w-full bg-blue-50 text-blue-600 text-xs py-2 rounded border border-blue-100 font-semibold group-hover:bg-blue-100 transition-colors disabled:opacity-50">
                            {isUploading ? "Uploading..." : "+ Upload New File"}
                        </button>
                    </div>

                    {config.file_path && !isUploading && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-[10px] text-gray-500 break-all font-mono border border-gray-100 flex justify-between items-center">
                            <span><span className="font-bold text-gray-700">Linked:</span> {config.file_path.split('/').pop()}</span>
                            {isLoadingContent && <span className="text-blue-500 animate-pulse">Loading...</span>}
                        </div>
                    )}
                </div>

                {/* 2. Target & Features */}
                {(headers.length > 0) ? (
                    <>
                        {/* TARGETS (Y) */}
                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <label className="block text-[10px] font-bold text-gray-500 uppercase">
                                    Target Columns (Y)
                                </label>
                                <span className="text-[10px] bg-red-50 text-red-600 px-1.5 rounded">
                                    {targets.length} selected
                                </span>
                            </div>
                            <div className="border rounded-md max-h-32 overflow-y-auto bg-gray-50 p-2 space-y-1">
                                {headers.map(h => {
                                    const isSelected = targets.includes(h);
                                    return (
                                        <label
                                            key={`t-${h}`}
                                            className="flex items-center gap-2 p-1 rounded text-xs cursor-pointer hover:bg-white transition-colors"
                                        >
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleTarget(h)}
                                                className="rounded text-red-600 focus:ring-red-500"
                                            />
                                            <span className={`truncate ${isSelected ? 'font-medium text-gray-900' : 'text-gray-600'}`}>
                                                {h}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        </div>

                        {/* FEATURES (X) */}
                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <label className="block text-[10px] font-bold text-gray-500 uppercase">
                                    Feature Columns (X)
                                </label>
                                <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 rounded">
                                    {features.length} selected
                                </span>
                            </div>

                            <div className="border rounded-md max-h-48 overflow-y-auto bg-gray-50 p-2 space-y-1">
                                {headers.map(h => {
                                    const isSelected = features.includes(h);
                                    return (
                                        <label
                                            key={`f-${h}`}
                                            className="flex items-center gap-2 p-1 rounded text-xs cursor-pointer hover:bg-white transition-colors"
                                        >
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleFeature(h)}
                                                className="rounded text-blue-600 focus:ring-blue-500"
                                            />
                                            <span className={`truncate ${isSelected ? 'font-medium text-gray-900' : 'text-gray-600'}`}>
                                                {h}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        </div>
                    </>
                ) : (
                    // Fallback message
                    config.file_path && !isLoadingContent && (
                        <div className="text-xs text-orange-600 bg-orange-50 p-3 rounded border border-orange-100">
                            <strong>No columns found.</strong>
                            <br />
                            Try re-uploading the file if you suspect it is corrupted or empty.
                        </div>
                    )
                )}

                {/* 3. Sequence Config (Always Visible) */}
                <div className="pt-4 border-t border-gray-100">
                    <label className="block text-[10px] font-bold text-gray-500 uppercase mb-2">
                        Sequence Length
                    </label>
                    <input
                        type="number"
                        min="1"
                        value={seqLen}
                        onChange={(e) => {
                            const val = parseInt(e.target.value);
                            onConfigChange({ ...config, sequence_length: isNaN(val) ? 1 : val });
                        }}
                        className="w-full text-xs border border-gray-300 rounded p-1.5 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    />
                    <p className="text-[10px] text-gray-400 mt-1 leading-tight">
                        <strong>1</strong> = Standard Tabular.<br />
                        <strong>&gt;1</strong> = Time Series (Rows per sample).
                    </p>
                </div>
            </div>
        </div>
    );
}
