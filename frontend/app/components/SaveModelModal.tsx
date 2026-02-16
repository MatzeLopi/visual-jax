"use client";

import React, { useState, useEffect } from 'react';

interface SaveModelModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, isPublic: boolean) => Promise<void>;
    isLoggedIn: boolean;
    initialName?: string;
}

export default function SaveModelModal({ isOpen, onClose, onSave, isLoggedIn, initialName = "" }: SaveModelModalProps) {
    const [name, setName] = useState(initialName);
    const [isPublic, setIsPublic] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState("");

    // Reset state when opening
    useEffect(() => {
        if (isOpen) {
            setName(initialName);
            setIsPublic(false);
            setIsSaving(false);
            setError("");
        }
    }, [isOpen, initialName]);

    if (!isOpen) return null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSaving(true);
        setError("");
        try {
            await onSave(name, isLoggedIn ? isPublic : true);
            onClose();
        } catch (err: any) {
            setError(err.message || "Failed to save");
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-md bg-white rounded-lg shadow-xl overflow-hidden border border-gray-200 animate-in fade-in zoom-in duration-200">
                <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                    <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wide">Save Model</h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    {error && (
                        <div className="p-2 bg-red-50 text-red-600 text-xs rounded border border-red-100">
                            {error}
                        </div>
                    )}

                    <div>
                        <label className="block text-xs font-bold text-gray-700 mb-1 uppercase tracking-wide">Model Name</label>
                        <input
                            type="text"
                            required
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-black focus:border-black transition-shadow"
                            placeholder="My Neural Network"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            autoFocus
                        />
                    </div>

                    {isLoggedIn && (
                        <div className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                id="isPublic"
                                className="rounded border-gray-300 text-black focus:ring-black cursor-pointer"
                                checked={isPublic}
                                onChange={(e) => setIsPublic(e.target.checked)}
                            />
                            <label htmlFor="isPublic" className="text-sm text-gray-700 cursor-pointer select-none">Make this model public</label>
                        </div>
                    )}

                    {!isLoggedIn && (
                        <p className="text-xs text-gray-500 italic bg-blue-50 p-2 rounded border border-blue-100 text-blue-700">
                            You are not logged in. This model will be saved as <strong>Public</strong>.
                        </p>
                    )}

                    <div className="pt-2 flex justify-end gap-3">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 text-xs font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                            disabled={isSaving}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={!name.trim() || isSaving}
                            className="px-4 py-2 text-xs font-medium text-white bg-black rounded-md hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                        >
                            {isSaving && <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>}
                            {isSaving ? "Saving..." : "Save Model"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
