"use client";
import { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useRouter } from 'next/navigation';
import api from '../lib/api';
import {
    Lock,
    LogOut,
    User as UserIcon,
    Shield,
    Menu,
    X,
} from 'lucide-react';

export default function Dashboard() {
    const { user, logout, isLoading } = useAuth();
    const router = useRouter();

    const [pwData, setPwData] = useState({ old_password: '', new_password: '' });
    const [status, setStatus] = useState({ type: '', msg: '' });
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        if (!isLoading && !user) router.push('/login');
    }, [user, isLoading, router]);

    const handlePasswordChange = async (e: React.FormEvent) => {
        e.preventDefault();
        setStatus({ type: '', msg: '' });
        setIsSubmitting(true);

        if (pwData.new_password.length < 8) {
            setStatus({ type: 'error', msg: 'Password must be 8+ chars' });
            setIsSubmitting(false);
            return;
        }

        try {
            await api.post('/users/me/update-password', pwData);
            setStatus({ type: 'success', msg: 'Password updated' });
            setPwData({ old_password: '', new_password: '' });
        } catch (err) {
            setStatus({ type: 'error', msg: 'Update failed. Check current password.' });
        } finally {
            setIsSubmitting(false);
        }
    };

    if (isLoading || !user) return <div className="flex min-h-screen items-center justify-center bg-gray-50 text-sm text-gray-500">Loading...</div>;

    return (
        <div className="min-h-screen bg-[#F9FAFB]">

            {/* --- NAVBAR --- */}
            <nav className="sticky top-0 z-30 border-b border-gray-200 bg-white/80 backdrop-blur-md">
                <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
                    <div className="flex h-16 justify-between items-center">
                        {/* Logo */}
                        <div className="flex items-center gap-2">
                            <div className="rounded bg-gray-900 p-1.5 text-white">
                                <Shield size={16} strokeWidth={3} />
                            </div>
                            <span className="text-sm font-bold text-gray-900 tracking-tight">User Dashboard</span>
                        </div>

                        {/* Desktop Nav Actions */}
                        <div className="hidden md:flex items-center gap-6">
                            {/* Navigation Links */}
                            <div className="flex items-center gap-4 text-sm font-medium">
                                <a href="/editor" className="text-gray-600 hover:text-gray-900 transition-colors">Editor</a>
                                <a href="/training" className="text-gray-600 hover:text-gray-900 transition-colors">Training</a>
                            </div>

                            {/* User Info with Separator */}
                            <div className="flex items-center gap-3 pr-4 border-r border-gray-200">
                                <div className="h-8 w-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center text-xs font-bold text-gray-600">
                                    {user.username?.charAt(0).toUpperCase()}
                                </div>
                                <div className="flex flex-col text-right">
                                    <span className="text-sm font-semibold text-gray-900 leading-none">{user.username}</span>
                                    <span className="text-xs text-gray-500 leading-none mt-1">{user.email}</span>
                                </div>
                            </div>

                            {/* Styled Logout Button */}
                            <button
                                onClick={logout}
                                className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 hover:text-red-600 transition-all"
                            >
                                <LogOut size={14} />
                                Sign out
                            </button>
                        </div>

                        {/* Mobile Menu Toggle */}
                        <button
                            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                            className="md:hidden text-gray-500"
                            aria-expanded={isMobileMenuOpen}
                            aria-controls="mobile-menu"
                            aria-label="Toggle navigation menu"
                        >
                            {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
                        </button>
                    </div>
                </div>

                {/* Mobile Dropdown */}
                {isMobileMenuOpen && (
                    <div id="mobile-menu" className="md:hidden border-t border-gray-100 bg-white px-4 py-3">
                        <div className="mb-4 flex flex-col gap-2">
                            <a href="/editor" className="block text-sm font-medium text-gray-700 py-2 border-b border-gray-100">Editor</a>
                            <a href="/training" className="block text-sm font-medium text-gray-700 py-2 border-b border-gray-100">Training</a>
                        </div>
                        <div className="mb-3 flex items-center gap-3">
                            <div className="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center text-xs font-bold text-gray-600">
                                {user.username?.charAt(0).toUpperCase()}
                            </div>
                            <div className="text-sm text-gray-500">
                                {user.email}
                            </div>
                        </div>
                        <button onClick={logout} className="flex w-full items-center gap-2 rounded-md bg-gray-50 px-3 py-2 text-sm font-medium text-gray-700 active:bg-gray-100">
                            <LogOut size={16} /> Sign Out
                        </button>
                    </div>
                )}
            </nav>

            {/* --- CONTENT --- */}
            <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 gap-6 md:grid-cols-12 items-stretch">
                    {/* 'items-stretch' ensures grid children match height */}

                    {/* PROFILE CARD (Left Side) */}
                    <div className="md:col-span-4 lg:col-span-3">
                        {/* h-full and flex-col ensure the card fills the height and we can push content to bottom */}
                        <div className="h-full flex flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
                            <div className="bg-gray-50 p-8 flex flex-col items-center border-b border-gray-100">
                                <div className="h-24 w-24 rounded-full bg-white border border-gray-200 flex items-center justify-center text-gray-400 mb-4 shadow-sm">
                                    <UserIcon size={40} strokeWidth={1.5} />
                                </div>
                                <h2 className="text-xl font-bold text-gray-900">{user.username}</h2>
                                <p className="text-xs text-gray-500">{user.email}</p>
                            </div>

                            {/* mt-auto pushes this section to the bottom of the card */}
                            <div className="p-4 mt-auto">
                                <div className="flex items-center justify-between px-4 py-3 text-sm text-gray-600 bg-gray-50 rounded-lg border border-gray-100">
                                    <span className="font-medium">Account Status</span>
                                    <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                                        Active
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* SETTINGS CARD (Right Side) */}
                    <div className="md:col-span-8 lg:col-span-9">
                        <div className="h-full rounded-xl border border-gray-200 bg-white shadow-sm">
                            <div className="border-b border-gray-100 px-6 py-4">
                                <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
                                    <Lock size={16} className="text-gray-400" />
                                    Security Settings
                                </h3>
                            </div>

                            <div className="p-6">
                                <form onSubmit={handlePasswordChange} className="max-w-lg space-y-5">
                                    {status.msg && (
                                        <div className={`text-sm px-3 py-2 rounded-md ${status.type === 'error' ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'
                                            }`}>
                                            {status.msg}
                                        </div>
                                    )}

                                    <div className="space-y-4">
                                        <div>
                                            <label htmlFor="old_password" className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
                                            <input
                                                id="old_password"
                                                type="password"
                                                required
                                                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-gray-900 focus:ring-gray-900 sm:text-sm py-2 px-3 border"
                                                value={pwData.old_password}
                                                onChange={(e) => setPwData({ ...pwData, old_password: e.target.value })}
                                            />
                                        </div>
                                        <div>
                                            <label htmlFor="new_password" className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                                            <input
                                                id="new_password"
                                                type="password"
                                                required
                                                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-gray-900 focus:ring-gray-900 sm:text-sm py-2 px-3 border"
                                                value={pwData.new_password}
                                                onChange={(e) => setPwData({ ...pwData, new_password: e.target.value })}
                                            />
                                        </div>
                                    </div>

                                    <div className="pt-4 flex justify-end">
                                        <button
                                            type="submit"
                                            disabled={isSubmitting}
                                            className="inline-flex items-center justify-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-gray-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gray-900 disabled:opacity-50 transition-colors"
                                        >
                                            {isSubmitting ? 'Updating...' : 'Update Password'}
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>

                </div>
            </main>
        </div>
    );
}