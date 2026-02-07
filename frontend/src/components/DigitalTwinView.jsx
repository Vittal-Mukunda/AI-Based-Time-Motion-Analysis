import React, { useRef, useEffect, useCallback } from 'react';

// MediaPipe hand landmark connections
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
];

const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = 720;

const COLORS = {
    rightHand: '#00FFFF',   // Cyan
    leftHand: '#FF00FF',    // Magenta
    board: '#00FF88'        // Green
};

function DigitalTwinView({ subscribe, getDataRef }) {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const lastDataRef = useRef(null);

    const drawGrid = useCallback((ctx) => {
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.08)';
        ctx.lineWidth = 1;
        for (let x = 0; x < CANVAS_WIDTH; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, CANVAS_HEIGHT);
            ctx.stroke();
        }
        for (let y = 0; y < CANVAS_HEIGHT; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(CANVAS_WIDTH, y);
            ctx.stroke();
        }
    }, []);

    const drawBoard = useCallback((ctx, boardZone) => {
        if (!boardZone) return;
        const { x1, y1, x2, y2 } = boardZone;

        ctx.strokeStyle = COLORS.board;
        ctx.lineWidth = 3;
        ctx.shadowColor = COLORS.board;
        ctx.shadowBlur = 15;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = COLORS.board;
        ctx.font = '600 14px Inter, Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('WORK ZONE', (x1 + x2) / 2, y1 - 12);
    }, []);

    const drawHand = useCallback((ctx, handLabel, handData) => {
        if (!handData.visible || handData.landmarks.length === 0) return;

        const color = handLabel === 'Left' ? COLORS.leftHand : COLORS.rightHand;
        const scaledLandmarks = handData.landmarks.map(([x, y]) => [
            x * CANVAS_WIDTH,
            y * CANVAS_HEIGHT
        ]);

        // Connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;

        for (const [start, end] of HAND_CONNECTIONS) {
            if (scaledLandmarks[start] && scaledLandmarks[end]) {
                ctx.beginPath();
                ctx.moveTo(scaledLandmarks[start][0], scaledLandmarks[start][1]);
                ctx.lineTo(scaledLandmarks[end][0], scaledLandmarks[end][1]);
                ctx.stroke();
            }
        }

        // Joints
        ctx.shadowBlur = 15;
        for (let i = 0; i < scaledLandmarks.length; i++) {
            const [x, y] = scaledLandmarks[i];
            const radius = [0, 4, 8, 12, 16, 20].includes(i) ? 8 : 5;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }
        ctx.shadowBlur = 0;

        // Hand label
        const [cx, cy] = handData.center;
        ctx.fillStyle = color;
        ctx.font = 'bold 18px Inter, Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${handLabel.toUpperCase()} HAND`, cx, cy - 50);

        ctx.font = '14px Inter, Roboto, sans-serif';
        ctx.fillStyle = '#aaaaaa';
        ctx.fillText(`State: ${handData.state}`, cx, cy - 30);
    }, []);

    const draw = useCallback((data) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Dark background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Grid
        drawGrid(ctx);

        if (!data) {
            ctx.fillStyle = '#1e90ff';
            ctx.font = '500 24px Inter, Roboto, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('DIGITAL TWIN • AWAITING DATA STREAM', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
            return;
        }

        // Board zone
        drawBoard(ctx, data.board_zone);

        // Hands
        for (const [handLabel, handData] of Object.entries(data.hands)) {
            drawHand(ctx, handLabel, handData);
        }

        // Stats overlay
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(15, 15, 180, 60);
        ctx.strokeStyle = '#1e90ff';
        ctx.lineWidth = 1;
        ctx.strokeRect(15, 15, 180, 60);

        ctx.fillStyle = '#1e90ff';
        ctx.font = '600 12px Inter, Roboto, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('MACHINE VISION', 25, 35);
        ctx.fillStyle = '#888888';
        ctx.font = '11px Inter, Roboto, sans-serif';
        ctx.fillText(`Frame: ${data.frame_id || 0}`, 25, 52);
        ctx.fillText(`Time: ${(data.video_time || 0).toFixed(2)}s`, 25, 67);

    }, [drawGrid, drawBoard, drawHand]);

    // Animation loop
    useEffect(() => {
        let isRunning = true;

        const animate = () => {
            if (!isRunning) return;
            const dataRef = getDataRef();
            if (dataRef.current !== lastDataRef.current) {
                draw(dataRef.current);
                lastDataRef.current = dataRef.current;
            }
            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            isRunning = false;
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [getDataRef, draw]);

    useEffect(() => {
        return subscribe(() => {
            lastDataRef.current = null;
        });
    }, [subscribe]);

    return (
        <div className="w-full h-full flex items-center justify-center">
            <canvas
                ref={canvasRef}
                width={CANVAS_WIDTH}
                height={CANVAS_HEIGHT}
                className="max-w-full max-h-full rounded-lg border border-gray-700"
                style={{
                    aspectRatio: `${CANVAS_WIDTH}/${CANVAS_HEIGHT}`,
                    imageRendering: 'crisp-edges'
                }}
            />
        </div>
    );
}

export default DigitalTwinView;
