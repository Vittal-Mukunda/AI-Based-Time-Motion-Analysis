import React, { useMemo, useState } from 'react';
import {
    LineChart, Line, AreaChart, Area,
    ScatterChart, Scatter, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { useDashboard } from '../context/DashboardContext';

// Colors
const ACCENT_CYAN = '#00FFFF';
const ACCENT_MAGENTA = '#FF00FF';
const ACCENT_GREEN = '#22c55e';
const ACCENT_RED = '#ef4444';
const ACCENT_BLUE = '#3b82f6';
const GRID_COLOR = '#334155';
const CARD_BG = 'rgba(30, 41, 59, 0.7)';

// Pie chart colors
const PIE_COLORS = {
    Operation: '#22c55e',  // Emerald
    Transport: '#f59e0b',  // Amber
    Delay: '#f43f5e'       // Rose
};

// Chart descriptions with EXACT FORMULAS
const CHART_INFO = {
    pathComplexity: "Formula: RMS(Deviation). Measures root mean square deviation from the optimal linear path. Score of 1.0 = perfect efficiency.",
    handVelocity: "Formula: v = Δd / Δt. Euclidean distance change per frame (px/sec). Peaks indicate rapid transport; valleys indicate precision work.",
    motionSymmetry: "Formula: |Left_Activity - Right_Activity|. Zero indicates perfect balance. Points on diagonal show balanced, simultaneous work.",
    timeBreakdown: "Tracks the ratio of time spent in Operation, Transport, and Delay states. Derived from backend state machine.",
    fatigue: "Formula: MovingAvg(Velocity, window=30s). Declining trend indicates fatigue. Rising line = increased effort or recovery.",
    motionPath: "A 'Spaghetti Diagram' visualizing the exact travel path of the hands across the workspace. Filtered with EMA smoothing."
};

// Info Icon with Tooltip
function InfoTooltip({ text }) {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <div className="relative inline-block ml-2">
            <button
                onMouseEnter={() => setIsVisible(true)}
                onMouseLeave={() => setIsVisible(false)}
                className="w-4 h-4 rounded-full bg-slate-600 hover:bg-slate-500 flex items-center justify-center text-[10px] text-slate-300 transition-colors"
                style={{ fontFamily: '"Times New Roman", serif' }}
            >
                i
            </button>
            {isVisible && (
                <div
                    className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 bg-white border-2 border-black rounded-lg shadow-xl text-xs text-black leading-relaxed"
                    style={{ fontFamily: '"Times New Roman", serif' }}
                >
                    {text}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-black"></div>
                </div>
            )}
        </div>
    );
}

// Glassmorphism Chart Card
function GlassCard({ title, info, children }) {
    return (
        <div
            className="rounded-xl p-4 flex flex-col h-full min-h-[240px] border border-slate-600/50"
            style={{
                background: CARD_BG,
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)'
            }}
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                    <h3 className="text-sm font-semibold text-white" style={{ fontFamily: '"Times New Roman", serif' }}>{title}</h3>
                    <InfoTooltip text={info} />
                </div>
            </div>
            <div className="flex-1 min-h-0">
                {children}
            </div>
        </div>
    );
}

// Control Bar Component
function ControlBar({ isRecording, onStart, onPause, onReset }) {
    return (
        <div
            className="flex items-center gap-4 mb-6 p-4 rounded-xl border border-slate-600/50"
            style={{
                background: CARD_BG,
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                fontFamily: '"Times New Roman", serif'
            }}
        >
            <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-slate-500'}`} />
                <span className={`text-sm font-medium ${isRecording ? 'text-red-400' : 'text-slate-400'}`}>
                    {isRecording ? 'RECORDING' : 'PAUSED'}
                </span>
            </div>

            <div className="flex-1" />

            {!isRecording ? (
                <button
                    onClick={onStart}
                    className="flex items-center gap-2 px-5 py-2.5 bg-green-600/80 hover:bg-green-500 text-white rounded-lg font-medium transition-all hover:scale-105 shadow-lg shadow-green-900/30"
                >
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                    </svg>
                    START
                </button>
            ) : (
                <button
                    onClick={onPause}
                    className="flex items-center gap-2 px-5 py-2.5 bg-yellow-600/80 hover:bg-yellow-500 text-white rounded-lg font-medium transition-all hover:scale-105 shadow-lg shadow-yellow-900/30"
                >
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                    </svg>
                    PAUSE
                </button>
            )}

            <button
                onClick={onReset}
                className="flex items-center gap-2 px-5 py-2.5 bg-slate-600/80 hover:bg-slate-500 text-white rounded-lg font-medium transition-all hover:scale-105 shadow-lg shadow-slate-900/30"
            >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
                </svg>
                RESET
            </button>
        </div>
    );
}

// Fixed tooltip style - WHITE background, BLACK text, BLACK border
const tooltipStyle = {
    backgroundColor: '#ffffff',
    border: '2px solid #000000',
    borderRadius: '8px',
    fontSize: 11,
    color: '#000000',
    fontFamily: '"Times New Roman", serif',
    boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
};

// Legend config for top-right position
const legendProps = {
    wrapperStyle: { fontSize: 10, fontFamily: '"Times New Roman", serif' },
    align: 'right',
    verticalAlign: 'top',
    iconSize: 8
};

// Axis tick style with Times New Roman
const axisTickStyle = { fill: '#64748b', fontSize: 10, fontFamily: '"Times New Roman", serif' };
const axisLabelStyle = { fill: '#94a3b8', fontSize: 10, fontFamily: '"Times New Roman", serif' };

function AnalyticsCommandCenter() {
    const { isRecording, sessionData, startRecording, pauseRecording, resetSession } = useDashboard();

    const {
        pathComplexity,
        velocityData,
        symmetryData,
        timeBreakdown,
        fatigueData,
        motionPathLeft,
        motionPathRight
    } = sessionData;

    // Get current video time for display
    const currentTime = velocityData.length > 0 ? velocityData[velocityData.length - 1].time : 0;

    // Calculate fatigue trend
    const fatigueTrend = useMemo(() => {
        if (fatigueData.length < 10) return 'stable';
        const recent = fatigueData.slice(-15);
        const firstHalf = recent.slice(0, 7).reduce((a, b) => a + parseFloat(b.fatigue || 0), 0) / 7;
        const secondHalf = recent.slice(-7).reduce((a, b) => a + parseFloat(b.fatigue || 0), 0) / 7;
        return secondHalf > firstHalf + 3 ? 'rising' : secondHalf < firstHalf - 3 ? 'falling' : 'stable';
    }, [fatigueData]);

    // Prepare pie chart data
    const pieData = useMemo(() => {
        const total = timeBreakdown.Operation + timeBreakdown.Transport + timeBreakdown.Delay;
        if (total === 0) return [];
        return [
            { name: 'Operation', value: timeBreakdown.Operation, fill: PIE_COLORS.Operation },
            { name: 'Transport', value: timeBreakdown.Transport, fill: PIE_COLORS.Transport },
            { name: 'Delay', value: timeBreakdown.Delay, fill: PIE_COLORS.Delay }
        ].filter(d => d.value > 0);
    }, [timeBreakdown]);

    return (
        <div
            className="h-full overflow-auto p-6"
            style={{
                background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
                fontFamily: '"Times New Roman", serif'
            }}
        >
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <h1 className="text-2xl font-bold text-white mb-1" style={{ fontFamily: '"Times New Roman", serif' }}>
                        Analytics Command Center
                    </h1>
                    <div className="flex items-center gap-4 text-sm text-slate-400" style={{ fontFamily: '"Times New Roman", serif' }}>
                        <span>Real-time motion analysis metrics</span>
                        <span className="text-slate-600">|</span>
                        <span className="text-cyan-400">Video Time: {currentTime}s</span>
                        <span className="text-slate-600">|</span>
                        <span>Data Points: {velocityData.length}</span>
                    </div>
                </div>

                {/* Control Bar */}
                <ControlBar
                    isRecording={isRecording}
                    onStart={startRecording}
                    onPause={pauseRecording}
                    onReset={resetSession}
                />

                {/* 2x3 Grid (6 charts - NO Jerk/Smoothness) */}
                <div className="grid grid-cols-3 gap-5 mb-5">

                    {/* Chart 1: Path Complexity (RMS) */}
                    <GlassCard title="Path Complexity" info={CHART_INFO.pathComplexity}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={pathComplexity} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                                <XAxis
                                    dataKey="time"
                                    type="number"
                                    domain={['dataMin', 'dataMax']}
                                    tick={axisTickStyle}
                                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -10, ...axisLabelStyle }}
                                />
                                <YAxis
                                    tick={axisTickStyle}
                                    domain={[1, 'auto']}
                                    label={{ value: 'Ratio', angle: -90, position: 'insideLeft', ...axisLabelStyle }}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend {...legendProps} />
                                <ReferenceLine y={1} stroke={ACCENT_GREEN} strokeDasharray="5 5" />
                                <Line
                                    type="monotone"
                                    dataKey="efficiency"
                                    name="Path Deviation"
                                    stroke={ACCENT_BLUE}
                                    strokeWidth={2}
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </GlassCard>

                    {/* Chart 2: Hand Velocity Profile */}
                    <GlassCard title="Hand Velocity Profile" info={CHART_INFO.handVelocity}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={velocityData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                                <XAxis
                                    dataKey="time"
                                    type="number"
                                    domain={['dataMin', 'dataMax']}
                                    tick={axisTickStyle}
                                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -10, ...axisLabelStyle }}
                                />
                                <YAxis
                                    tick={axisTickStyle}
                                    label={{ value: 'px/sec', angle: -90, position: 'insideLeft', ...axisLabelStyle }}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend {...legendProps} />
                                <Line type="monotone" dataKey="left" name="Left" stroke={ACCENT_MAGENTA} strokeWidth={2} dot={false} isAnimationActive={false} />
                                <Line type="monotone" dataKey="right" name="Right" stroke={ACCENT_CYAN} strokeWidth={2} dot={false} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </GlassCard>

                    {/* Chart 3: Motion Symmetry */}
                    <GlassCard title="Motion Symmetry" info={CHART_INFO.motionSymmetry}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                                <XAxis
                                    type="number"
                                    dataKey="left"
                                    tick={axisTickStyle}
                                    domain={[0, 'auto']}
                                    label={{ value: 'Left', position: 'insideBottom', offset: -10, ...axisLabelStyle }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="right"
                                    tick={axisTickStyle}
                                    domain={[0, 'auto']}
                                    label={{ value: 'Right', angle: -90, position: 'insideLeft', ...axisLabelStyle }}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend {...legendProps} />
                                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke={ACCENT_GREEN} strokeDasharray="5 5" />
                                <Scatter name="Balance" data={symmetryData} fill={ACCENT_BLUE} opacity={0.7} isAnimationActive={false} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </GlassCard>

                    {/* Chart 4: Time Breakdown (Donut Chart) */}
                    <GlassCard title="Time Breakdown" info={CHART_INFO.timeBreakdown}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart margin={{ top: 5, right: 80, left: 5, bottom: 5 }}>
                                <Pie
                                    data={pieData}
                                    cx="40%"
                                    cy="50%"
                                    innerRadius={40}
                                    outerRadius={65}
                                    paddingAngle={2}
                                    dataKey="value"
                                    isAnimationActive={false}
                                    label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                                    labelLine={false}
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.fill} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={tooltipStyle}
                                    formatter={(value) => `${value.toFixed(1)}s`}
                                />
                                <Legend
                                    layout="vertical"
                                    align="right"
                                    verticalAlign="middle"
                                    wrapperStyle={{ fontSize: 11, right: 0, fontFamily: '"Times New Roman", serif' }}
                                    formatter={(value) => <span style={{ color: '#94a3b8' }}>{value}</span>}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </GlassCard>

                    {/* Chart 5: Fatigue Index */}
                    <GlassCard title={`Fatigue Index (${fatigueTrend})`} info={CHART_INFO.fatigue}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={fatigueData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                                <defs>
                                    <linearGradient id="fatigueGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={fatigueTrend === 'rising' ? ACCENT_RED : ACCENT_GREEN} stopOpacity={0.6} />
                                        <stop offset="95%" stopColor={fatigueTrend === 'rising' ? ACCENT_RED : ACCENT_GREEN} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                                <XAxis
                                    dataKey="time"
                                    type="number"
                                    domain={['dataMin', 'dataMax']}
                                    tick={axisTickStyle}
                                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -10, ...axisLabelStyle }}
                                />
                                <YAxis
                                    tick={axisTickStyle}
                                    domain={[0, 100]}
                                    label={{ value: 'Score', angle: -90, position: 'insideLeft', ...axisLabelStyle }}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend {...legendProps} />
                                <Area
                                    type="monotone"
                                    dataKey="fatigue"
                                    name="Fatigue"
                                    stroke={fatigueTrend === 'rising' ? ACCENT_RED : ACCENT_GREEN}
                                    fill="url(#fatigueGradient)"
                                    strokeWidth={2}
                                    isAnimationActive={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </GlassCard>

                    {/* Chart 6: Motion Path Trace (in grid, not full width) */}
                    <GlassCard title="Motion Path Trace" info={CHART_INFO.motionPath}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    domain={[0, 1280]}
                                    tick={axisTickStyle}
                                    label={{ value: 'X (px)', position: 'insideBottom', offset: -10, ...axisLabelStyle }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    domain={[0, 720]}
                                    tick={axisTickStyle}
                                    reversed
                                    label={{ value: 'Y', angle: -90, position: 'insideLeft', ...axisLabelStyle }}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend {...legendProps} />
                                <Scatter
                                    name="Left"
                                    data={motionPathLeft}
                                    fill={ACCENT_MAGENTA}
                                    line={{ stroke: ACCENT_MAGENTA, strokeWidth: 1 }}
                                    lineType="joint"
                                    shape="circle"
                                    opacity={0.5}
                                    isAnimationActive={false}
                                />
                                <Scatter
                                    name="Right"
                                    data={motionPathRight}
                                    fill={ACCENT_CYAN}
                                    line={{ stroke: ACCENT_CYAN, strokeWidth: 1 }}
                                    lineType="joint"
                                    shape="circle"
                                    opacity={0.5}
                                    isAnimationActive={false}
                                />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </GlassCard>
                </div>

                {/* Status Bar */}
                <div className="mt-5 flex items-center justify-center gap-6 text-xs text-slate-400" style={{ fontFamily: '"Times New Roman", serif' }}>
                    <span className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ACCENT_CYAN }}></span> Right Hand
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ACCENT_MAGENTA }}></span> Left Hand
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ACCENT_GREEN }}></span> Good
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ACCENT_RED }}></span> Warning
                    </span>
                </div>
            </div>
        </div>
    );
}

export default AnalyticsCommandCenter;
