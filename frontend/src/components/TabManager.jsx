import React, { useState } from 'react';

const TABS = [
    { id: 'live', label: 'Live Operations', icon: '📡' },
    { id: 'twin', label: 'Digital Twin', icon: '🔲' },
    { id: 'analytics', label: 'Analytics Suite', icon: '📊' }
];

function TabManager({ activeTab, onTabChange }) {
    return (
        <nav className="tab-nav flex items-center gap-1 bg-exec-card border-b border-exec-border px-4">
            {TABS.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={`tab-button relative px-5 py-3 text-sm font-medium transition-all duration-200 ${activeTab === tab.id
                            ? 'text-exec-blue'
                            : 'text-gray-400 hover:text-gray-200'
                        }`}
                >
                    <span className="flex items-center gap-2">
                        <span className="text-base">{tab.icon}</span>
                        <span>{tab.label}</span>
                    </span>

                    {/* Active indicator */}
                    {activeTab === tab.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-exec-blue rounded-t" />
                    )}
                </button>
            ))}
        </nav>
    );
}

export default TabManager;
