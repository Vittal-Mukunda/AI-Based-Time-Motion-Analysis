import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

const DashboardContext = createContext(null);

// Initial empty state for all chart data
const initialSessionData = {
    pathComplexity: [],
    velocityData: [],
    symmetryData: [],
    jerkData: [],
    timeBreakdown: { Operation: 0, Transport: 0, Delay: 0 },
    fatigueData: [],
    motionPathLeft: [],
    motionPathRight: []
};

// Chart update interval (200ms = 5 FPS for charts)
const CHART_UPDATE_INTERVAL = 200;

export function DashboardProvider({ children, subscribe, getDataRef }) {
    // Session control state
    const [isRecording, setIsRecording] = useState(false);
    const [sessionData, setSessionData] = useState(initialSessionData);

    // Refs for throttling - store latest data without triggering re-renders
    const latestDataRef = useRef(null);
    const prevVelocityRef = useRef({ left: 0, right: 0 });
    const prevPositionRef = useRef({ left: null, right: null });
    const totalDistanceRef = useRef({ left: 0, right: 0 });
    const startPositionRef = useRef({ left: null, right: null });
    const updateIntervalRef = useRef(null);

    // Start recording
    const startRecording = useCallback(() => {
        setIsRecording(true);
    }, []);

    // Pause recording
    const pauseRecording = useCallback(() => {
        setIsRecording(false);
    }, []);

    // Reset session (clears frontend data only)
    const resetSession = useCallback(() => {
        setSessionData(initialSessionData);
        prevVelocityRef.current = { left: 0, right: 0 };
        prevPositionRef.current = { left: null, right: null };
        totalDistanceRef.current = { left: 0, right: 0 };
        startPositionRef.current = { left: null, right: null };
    }, []);

    // Subscribe to WebSocket data (stores in ref, does NOT update state directly)
    useEffect(() => {
        if (!subscribe) return;

        const unsubscribe = subscribe((data) => {
            // Handle RESET_CONFIRMED event from backend
            if (data?.event === 'RESET_CONFIRMED') {
                resetSession();
                return;
            }
            // Always store latest data in ref (instant, no re-render)
            latestDataRef.current = data;
        });

        return unsubscribe;
    }, [subscribe, resetSession]);

    // Throttled chart update loop (5 FPS = every 200ms)
    useEffect(() => {
        if (!isRecording) {
            // Clear interval when paused
            if (updateIntervalRef.current) {
                clearInterval(updateIntervalRef.current);
                updateIntervalRef.current = null;
            }
            return;
        }

        // Start throttled update loop
        updateIntervalRef.current = setInterval(() => {
            const data = latestDataRef.current;
            if (!data) return;

            // Use video_time from WebSocket (NOT Date.now())
            const videoTime = data.video_time || 0;
            const time = parseFloat(videoTime.toFixed(1));

            // Extract hand data
            const leftHand = data.hands?.Left || { velocity: 0, visible: false, center: [0, 0], state: 'Delay' };
            const rightHand = data.hands?.Right || { velocity: 0, visible: false, center: [0, 0], state: 'Delay' };

            // Get counters from backend for time breakdown
            const counters = data.counters || {};

            setSessionData(prev => {
                const MAX_POINTS = 150;
                const newData = { ...prev };

                // 1. Path Complexity (RMS deviation) - sqrt(deviation^2)
                // Track actual distance vs straight line distance
                if (leftHand.center && leftHand.visible) {
                    const currentPos = { x: leftHand.center[0], y: leftHand.center[1] };
                    if (prevPositionRef.current.left) {
                        const dx = currentPos.x - prevPositionRef.current.left.x;
                        const dy = currentPos.y - prevPositionRef.current.left.y;
                        totalDistanceRef.current.left += Math.sqrt(dx * dx + dy * dy);
                    }
                    if (!startPositionRef.current.left) {
                        startPositionRef.current.left = currentPos;
                    }
                    prevPositionRef.current.left = currentPos;
                }
                if (rightHand.center && rightHand.visible) {
                    const currentPos = { x: rightHand.center[0], y: rightHand.center[1] };
                    if (prevPositionRef.current.right) {
                        const dx = currentPos.x - prevPositionRef.current.right.x;
                        const dy = currentPos.y - prevPositionRef.current.right.y;
                        totalDistanceRef.current.right += Math.sqrt(dx * dx + dy * dy);
                    }
                    if (!startPositionRef.current.right) {
                        startPositionRef.current.right = currentPos;
                    }
                    prevPositionRef.current.right = currentPos;
                }

                // Calculate path complexity ratio (always positive, baseline 1.0)
                let complexityRatio = 1.0;
                if (startPositionRef.current.left && prevPositionRef.current.left) {
                    const straightLineLeft = Math.sqrt(
                        Math.pow(prevPositionRef.current.left.x - startPositionRef.current.left.x, 2) +
                        Math.pow(prevPositionRef.current.left.y - startPositionRef.current.left.y, 2)
                    );
                    if (straightLineLeft > 10) {
                        const deviation = totalDistanceRef.current.left / straightLineLeft;
                        complexityRatio = Math.sqrt(deviation * deviation); // RMS: sqrt(deviation^2)
                    }
                }
                newData.pathComplexity = [...prev.pathComplexity.slice(-MAX_POINTS + 1), {
                    time,
                    efficiency: parseFloat(Math.max(1.0, complexityRatio).toFixed(2))
                }];

                // 2. Hand Velocity Profile - use video_time as X-axis
                newData.velocityData = [...prev.velocityData.slice(-MAX_POINTS + 1), {
                    time,
                    left: parseFloat((leftHand.velocity || 0).toFixed(1)),
                    right: parseFloat((rightHand.velocity || 0).toFixed(1))
                }];

                // 3. Motion Symmetry (scatter)
                if (leftHand.visible && rightHand.visible) {
                    newData.symmetryData = [...prev.symmetryData.slice(-100), {
                        left: leftHand.velocity || 0,
                        right: rightHand.velocity || 0
                    }];
                }

                // 4. Jerk Metric (rate of velocity change) - with null check
                const leftVel = leftHand.velocity || 0;
                const rightVel = rightHand.velocity || 0;
                const leftJerk = Math.abs(leftVel - prevVelocityRef.current.left);
                const rightJerk = Math.abs(rightVel - prevVelocityRef.current.right);
                prevVelocityRef.current = { left: leftVel, right: rightVel };

                const jerkValue = ((leftJerk + rightJerk) / 2) || 0; // Null check: || 0
                newData.jerkData = [...prev.jerkData.slice(-30), {
                    time,
                    jerk: parseFloat(jerkValue.toFixed(1))
                }];

                // 5. Time Breakdown (Donut Chart) - from backend counters
                const opTime = (counters.Left_Operation || 0) + (counters.Right_Operation || 0);
                const transTime = (counters.Left_Transport || 0) + (counters.Right_Transport || 0);
                const delayTime = (counters.Left_Delay || 0) + (counters.Right_Delay || 0);
                newData.timeBreakdown = {
                    Operation: parseFloat(opTime.toFixed(1)),
                    Transport: parseFloat(transTime.toFixed(1)),
                    Delay: parseFloat(delayTime.toFixed(1))
                };

                // 6. Fatigue Index (rolling average velocity trend)
                const currentVel = (leftVel + rightVel) / 2;
                const recentVelocities = prev.fatigueData.slice(-20).map(d => parseFloat(d.velocity) || 0);
                recentVelocities.push(currentVel);
                const avgRecent = recentVelocities.reduce((a, b) => a + b, 0) / recentVelocities.length;

                // Fatigue increases as velocity drops over time
                const baselineFatigue = 30;
                const fatigue = baselineFatigue + (50 - avgRecent) * 0.5 + (videoTime * 0.3);

                newData.fatigueData = [...prev.fatigueData.slice(-MAX_POINTS + 1), {
                    time,
                    fatigue: parseFloat(Math.max(0, Math.min(100, fatigue)).toFixed(1)),
                    velocity: parseFloat(currentVel.toFixed(1))
                }];

                // 7. Motion Path Trace (spaghetti diagram)
                if (leftHand.center && leftHand.visible) {
                    newData.motionPathLeft = [...prev.motionPathLeft.slice(-300), {
                        x: leftHand.center[0],
                        y: leftHand.center[1]
                    }];
                }
                if (rightHand.center && rightHand.visible) {
                    newData.motionPathRight = [...prev.motionPathRight.slice(-300), {
                        x: rightHand.center[0],
                        y: rightHand.center[1]
                    }];
                }

                return newData;
            });
        }, CHART_UPDATE_INTERVAL);

        return () => {
            if (updateIntervalRef.current) {
                clearInterval(updateIntervalRef.current);
                updateIntervalRef.current = null;
            }
        };
    }, [isRecording]);

    const value = {
        isRecording,
        sessionData,
        startRecording,
        pauseRecording,
        resetSession
    };

    return (
        <DashboardContext.Provider value={value}>
            {children}
        </DashboardContext.Provider>
    );
}

export function useDashboard() {
    const context = useContext(DashboardContext);
    if (!context) {
        throw new Error('useDashboard must be used within DashboardProvider');
    }
    return context;
}

export default DashboardContext;
