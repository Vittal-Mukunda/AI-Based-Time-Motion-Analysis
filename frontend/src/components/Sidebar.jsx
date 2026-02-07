import React, { useState, useEffect } from 'react';

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function StatCard({ title, value, unit, color }) {
    const colorClasses = {
        cyan: 'text-cyber-cyan border-cyber-cyan',
        green: 'text-cyber-green border-cyber-green',
        yellow: 'text-cyber-yellow border-cyber-yellow',
        gray: 'text-cyber-gray border-cyber-gray',
        blue: 'text-exec-blue border-exec-blue'
    };

    return (
        <div className={`cyber-card p-3 border-l-4 ${colorClasses[color] || colorClasses.blue}`}>
            <p className="text-xs text-gray-500 uppercase mb-1">{title}</p>
            <div className="flex items-baseline gap-1">
                <span className={`text-2xl font-bold ${colorClasses[color]?.split(' ')[0]}`}>
                    {value}
                </span>
                {unit && <span className="text-xs text-gray-500">{unit}</span>}
            </div>
        </div>
    );
}

function HandPanel({ label, data, counters }) {
    if (!data) return null;

    const stateColors = {
        Operation: 'green',
        Transport: 'yellow',
        Delay: 'gray'
    };

    const stateBgColors = {
        Operation: 'bg-cyber-green/20 border-cyber-green',
        Transport: 'bg-cyber-yellow/20 border-cyber-yellow',
        Delay: 'bg-cyber-gray/20 border-cyber-gray'
    };

    const totalTime = (counters?.[`${label}_Operation`] || 0) +
        (counters?.[`${label}_Transport`] || 0) +
        (counters?.[`${label}_Delay`] || 0);

    return (
        <div className="cyber-card p-4">
            <div className="flex items-center justify-between mb-4">
                <h3 className={`text-lg font-semibold ${label === 'Left' ? 'text-cyber-magenta' : 'text-cyber-cyan'}`}>
                    {label} Hand
                </h3>
                <div className={`px-3 py-1 rounded border ${stateBgColors[data.state]}`}>
                    <span className={`text-sm font-medium ${stateColors[data.state] === 'green' ? 'text-cyber-green' : stateColors[data.state] === 'yellow' ? 'text-cyber-yellow' : 'text-cyber-gray'}`}>
                        {data.state}
                    </span>
                </div>
            </div>

            {data.visible ? (
                <>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                        <StatCard
                            title="Velocity"
                            value={data.velocity?.toFixed(1) || '0.0'}
                            unit="px/f"
                            color={stateColors[data.state]}
                        />
                        <StatCard
                            title="Fingers"
                            value={data.fingers_in_zone || 0}
                            unit="in zone"
                            color={data.fingers_in_zone > 0 ? 'blue' : 'gray'}
                        />
                    </div>

                    {/* Time breakdown */}
                    <div className="space-y-2">
                        <p className="text-xs text-gray-500 uppercase">Time Breakdown</p>

                        {/* Operation */}
                        <div className="flex items-center gap-2">
                            <span className="text-xs w-16 text-cyber-green">Operation</span>
                            <div className="flex-1 h-2 bg-exec-bg rounded overflow-hidden">
                                <div
                                    className="h-full bg-cyber-green transition-all duration-300"
                                    style={{ width: `${totalTime > 0 ? ((counters?.[`${label}_Operation`] || 0) / totalTime) * 100 : 0}%` }}
                                />
                            </div>
                            <span className="text-xs text-gray-400 w-12 text-right">
                                {formatTime(counters?.[`${label}_Operation`] || 0)}
                            </span>
                        </div>

                        {/* Transport */}
                        <div className="flex items-center gap-2">
                            <span className="text-xs w-16 text-cyber-yellow">Transport</span>
                            <div className="flex-1 h-2 bg-exec-bg rounded overflow-hidden">
                                <div
                                    className="h-full bg-cyber-yellow transition-all duration-300"
                                    style={{ width: `${totalTime > 0 ? ((counters?.[`${label}_Transport`] || 0) / totalTime) * 100 : 0}%` }}
                                />
                            </div>
                            <span className="text-xs text-gray-400 w-12 text-right">
                                {formatTime(counters?.[`${label}_Transport`] || 0)}
                            </span>
                        </div>

                        {/* Delay */}
                        <div className="flex items-center gap-2">
                            <span className="text-xs w-16 text-cyber-gray">Delay</span>
                            <div className="flex-1 h-2 bg-exec-bg rounded overflow-hidden">
                                <div
                                    className="h-full bg-cyber-gray transition-all duration-300"
                                    style={{ width: `${totalTime > 0 ? ((counters?.[`${label}_Delay`] || 0) / totalTime) * 100 : 0}%` }}
                                />
                            </div>
                            <span className="text-xs text-gray-400 w-12 text-right">
                                {formatTime(counters?.[`${label}_Delay`] || 0)}
                            </span>
                        </div>
                    </div>
                </>
            ) : (
                <div className="text-center py-6 text-gray-500">
                    <p>Not detected</p>
                </div>
            )}
        </div>
    );
}

function Sidebar({ dataRef, subscribe }) {
    const [data, setData] = useState(null);

    useEffect(() => {
        // Update sidebar at reduced rate for performance
        const interval = setInterval(() => {
            if (dataRef.current) {
                setData({ ...dataRef.current });
            }
        }, 250);

        return () => clearInterval(interval);
    }, [dataRef]);

    return (
        <aside className="w-80 bg-exec-card border-l border-exec-border p-4 overflow-y-auto">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-4">Real-time Analytics</h2>

            <div className="space-y-4">
                <HandPanel
                    label="Left"
                    data={data?.hands?.Left}
                    counters={data?.counters}
                />
                <HandPanel
                    label="Right"
                    data={data?.hands?.Right}
                    counters={data?.counters}
                />

                {/* Video Time */}
                <div className="cyber-card p-4">
                    <p className="text-xs text-gray-500 uppercase mb-2">Video Progress</p>
                    <div className="text-3xl font-bold text-exec-blue text-center">
                        {data?.video_time ? formatTime(data.video_time) : '0:00'}
                    </div>
                </div>

                {/* Alerts */}
                {data?.alerts?.length > 0 && (
                    <div className="cyber-card p-4 border-l-4 border-cyber-magenta">
                        <p className="text-xs text-gray-500 uppercase mb-2">Alerts</p>
                        {data.alerts.map((alert, i) => (
                            <div key={i} className="text-sm text-cyber-magenta">
                                ⚠ {alert}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </aside>
    );
}

export default Sidebar;
