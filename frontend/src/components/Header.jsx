import React from 'react';

function Header({ isConnected, fps, frameId }) {
    return (
        <header className="h-14 bg-exec-card border-b border-exec-border flex items-center justify-between px-6">
            {/* Logo */}
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-exec-blue to-exec-accent flex items-center justify-center">
                    <span className="text-white font-bold text-lg">K</span>
                </div>
                <div>
                    <h1 className="text-lg font-semibold text-exec-blue">
                        KINETIC ANALYTICS
                    </h1>
                    <p className="text-xs text-gray-500">Executive Dashboard • v2.0</p>
                </div>
            </div>

            {/* Center Stats */}
            <div className="flex items-center gap-8">
                <div className="text-center">
                    <p className="text-xs text-gray-500 uppercase">FPS</p>
                    <p className="text-xl font-bold text-exec-blue">{fps?.toFixed(1) || '0.0'}</p>
                </div>
                <div className="text-center">
                    <p className="text-xs text-gray-500 uppercase">Frame</p>
                    <p className="text-xl font-bold text-white">{frameId || 0}</p>
                </div>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-3">
                <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'status-connected' : 'status-disconnected'}`} />
                <span className={`text-sm font-medium ${isConnected ? 'text-cyber-green' : 'text-red-400'}`}>
                    {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
            </div>
        </header>
    );
}

export default Header;
