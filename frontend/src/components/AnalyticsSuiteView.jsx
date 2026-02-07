import React, { useState, useEffect, useRef } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    RadialBarChart,
    RadialBar
} from 'recharts';

const MAX_DATA_POINTS = 30;

function EfficiencyGauge({ value }) {
    const data = [{ name: 'Efficiency', value: value, fill: '#1e90ff' }];

    return (
        <div className="bg-exec-card rounded-lg p-4 border border-exec-border">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Efficiency Score</h3>
            <div className="relative h-48">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                        cx="50%"
                        cy="50%"
                        innerRadius="60%"
                        outerRadius="90%"
                        startAngle={180}
                        endAngle={0}
                        data={data}
                    >
                        <RadialBar
                            background={{ fill: '#1a2332' }}
                            dataKey="value"
                            cornerRadius={10}
                        />
                    </RadialBarChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center pt-4">
                    <span className="text-4xl font-bold text-exec-blue">{value}%</span>
                    <span className="text-xs text-gray-500 mt-1">
                        {value >= 80 ? 'Excellent' : value >= 60 ? 'Good' : value >= 40 ? 'Average' : 'Needs Improvement'}
                    </span>
                </div>
            </div>
        </div>
    );
}

function StatCard({ title, value, unit, trend }) {
    return (
        <div className="bg-exec-card rounded-lg p-4 border border-exec-border">
            <p className="text-xs font-medium text-gray-400 uppercase mb-1">{title}</p>
            <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-white">{value}</span>
                {unit && <span className="text-sm text-gray-500">{unit}</span>}
            </div>
            {trend !== undefined && (
                <p className={`text-xs mt-1 ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}% from avg
                </p>
            )}
        </div>
    );
}

function AnalyticsSuiteView({ subscribe, getDataRef }) {
    const [cycleData, setCycleData] = useState([]);
    const [efficiency, setEfficiency] = useState(0);
    const [stats, setStats] = useState({
        avgCycleTime: 0,
        totalOperations: 0,
        transportTime: 0,
        delayTime: 0
    });
    const lastUpdateRef = useRef(Date.now());

    useEffect(() => {
        const unsubscribe = subscribe((data) => {
            if (!data || !data.counters) return;

            const now = Date.now();
            // Update at most every 500ms for smooth charts
            if (now - lastUpdateRef.current < 500) return;
            lastUpdateRef.current = now;

            const leftOp = data.counters.Left_Operation || 0;
            const leftTrans = data.counters.Left_Transport || 0;
            const leftDelay = data.counters.Left_Delay || 0;
            const rightOp = data.counters.Right_Operation || 0;
            const rightTrans = data.counters.Right_Transport || 0;
            const rightDelay = data.counters.Right_Delay || 0;

            const totalOp = leftOp + rightOp;
            const totalTrans = leftTrans + rightTrans;
            const totalDelay = leftDelay + rightDelay;
            const totalTime = totalOp + totalTrans + totalDelay;

            // Calculate efficiency (operation time / total time)
            const eff = totalTime > 0 ? Math.round((totalOp / totalTime) * 100) : 0;
            setEfficiency(eff);

            // Update stats
            setStats({
                avgCycleTime: totalTime > 0 ? (totalTime / Math.max(1, data.frame_id / 30)).toFixed(1) : 0,
                totalOperations: Math.round(totalOp),
                transportTime: Math.round(totalTrans),
                delayTime: Math.round(totalDelay)
            });

            // Add to cycle data chart
            setCycleData(prev => {
                const newPoint = {
                    time: (data.video_time || 0).toFixed(1),
                    cycleTime: totalTime > 0 ? (totalOp + totalTrans) / 10 : 0,
                    efficiency: eff
                };
                const updated = [...prev, newPoint];
                return updated.slice(-MAX_DATA_POINTS);
            });
        });

        return unsubscribe;
    }, [subscribe]);

    return (
        <div className="h-full overflow-auto p-6">
            <div className="max-w-6xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-xl font-semibold text-white">Analytics Dashboard</h2>
                        <p className="text-sm text-gray-400">Real-time performance metrics</p>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span className="inline-block w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                        Live Data
                    </div>
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-4 gap-4">
                    <StatCard title="Avg Cycle Time" value={stats.avgCycleTime} unit="sec" />
                    <StatCard title="Operation Time" value={stats.totalOperations} unit="sec" trend={5} />
                    <StatCard title="Transport Time" value={stats.transportTime} unit="sec" trend={-3} />
                    <StatCard title="Delay Time" value={stats.delayTime} unit="sec" trend={-8} />
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-3 gap-6">
                    {/* Cycle Time Chart */}
                    <div className="col-span-2 bg-exec-card rounded-lg p-4 border border-exec-border">
                        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Cycle Time vs Time</h3>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={cycleData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3a4a" />
                                    <XAxis
                                        dataKey="time"
                                        stroke="#64748b"
                                        tick={{ fill: '#64748b', fontSize: 11 }}
                                        label={{ value: 'Time (s)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }}
                                    />
                                    <YAxis
                                        stroke="#64748b"
                                        tick={{ fill: '#64748b', fontSize: 11 }}
                                        label={{ value: 'Cycle Time', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#0d1b2a',
                                            border: '1px solid #1e3a5f',
                                            borderRadius: '8px',
                                            color: '#fff'
                                        }}
                                    />
                                    <Legend wrapperStyle={{ color: '#64748b', fontSize: 12 }} />
                                    <Line
                                        type="monotone"
                                        dataKey="cycleTime"
                                        name="Cycle Time"
                                        stroke="#1e90ff"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="efficiency"
                                        name="Efficiency %"
                                        stroke="#00ff88"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Efficiency Gauge */}
                    <EfficiencyGauge value={efficiency} />
                </div>

                {/* Legend */}
                <div className="flex items-center justify-center gap-8 text-sm text-gray-400">
                    <div className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded-full bg-exec-blue"></span>
                        Cycle Time
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#00ff88' }}></span>
                        Efficiency
                    </div>
                </div>
            </div>
        </div>
    );
}

export default AnalyticsSuiteView;
