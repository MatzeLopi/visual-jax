"use client";
import React, { useState, useEffect } from 'react';
import api from '../lib/api';
import { TrainParams } from '../types';

interface Props {
    params: TrainParams;
    onChange: (params: TrainParams) => void;
    onClose: () => void;
    onStart: () => void;
}

export default function TrainingConfig({ params, onChange, onClose, onStart }: Props) {
    const [lossOptions, setLossOptions] = useState<string[]>([]);
    const [metricOptions, setMetricOptions] = useState<string[]>([]);

    useEffect(() => {
        api.get('/compiler/loss-types')
            .then(res => {
                setLossOptions(res.data);
                // Set default loss if not set
                if (res.data.length > 0 && !params.loss.type) {
                    onChange({ ...params, loss: { type: res.data[0] } });
                }
            })
            .catch(console.error);

        api.get('/compiler/metric-types')
            .then(res => setMetricOptions(res.data))
            .catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty dependency array means run once on mount

    const handleChange = <K extends keyof TrainParams>(field: K, value: TrainParams[K]) => {
        onChange({ ...params, [field]: value });
    };

    const handleMetricChange = (metric: string) => {
        let currentMetrics = params.metrics || [];
        const exists = currentMetrics.some(m => m.type === metric);
        if (exists) {
            currentMetrics = currentMetrics.filter(m => m.type !== metric);
        } else {
            currentMetrics = [...currentMetrics, { type: metric }];
        }
        handleChange('metrics', currentMetrics.length > 0 ? currentMetrics : null);
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-96 shadow-xl">
                <h2 className="text-lg font-bold mb-4 text-gray-900">Training Configuration</h2>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-1 text-gray-700">Epochs</label>
                        <input
                            type="number"
                            min="1"
                            value={params.epochs}
                            onChange={(e) => handleChange('epochs', parseInt(e.target.value) || 1)}
                            className="w-full border border-gray-300 rounded p-2 text-sm text-gray-900"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1 text-gray-700">Batch Size</label>
                        <input
                            type="number"
                            min="1"
                            value={params.batchsize}
                            onChange={(e) => handleChange('batchsize', parseInt(e.target.value) || 1)}
                            className="w-full border border-gray-300 rounded p-2 text-sm text-gray-900"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1 text-gray-700">Loss Function</label>
                        <select
                            value={params.loss.type}
                            onChange={(e) => handleChange('loss', { type: e.target.value })}
                            className="w-full border border-gray-300 rounded p-2 text-sm text-gray-900 bg-white"
                        >
                            {lossOptions.length === 0 && <option value="">Loading...</option>}
                            {lossOptions.map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1 text-gray-700">Metrics</label>
                        <div className="space-y-1 max-h-32 overflow-y-auto border border-gray-300 rounded p-2 bg-gray-50">
                            {metricOptions.length === 0 && <span className="text-xs text-gray-500">Loading...</span>}
                            {metricOptions.map(opt => (
                                <label key={opt} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={params.metrics?.some(m => m.type === opt) || false}
                                        onChange={() => handleMetricChange(opt)}
                                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                    />
                                    {opt}
                                </label>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="mt-6 flex justify-end gap-2">
                    <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded border border-gray-300">Cancel</button>
                    <button onClick={onStart} className="px-4 py-2 text-sm bg-black text-white hover:bg-gray-800 rounded font-medium">Start Training</button>
                </div>
            </div>
        </div>
    );
}
