import React, { useRef, useEffect, useCallback } from 'react';

// MediaPipe hand landmark connections
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],     // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]           // Palm
];

const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = 720;

// Executive color palette
const COLORS = {
    rightHand: { main: '#00FFFF', glow: 'rgba(0, 255, 255, 0.5)' },  // Cyan
    leftHand: { main: '#FF00FF', glow: 'rgba(255, 0, 255, 0.5)' },   // Magenta
    board: { fill: 'rgba(0, 255, 136, 0.15)', stroke: '#00FF88' },   // Green
    states: {
        Operation: '#00FF88',
        Transport: '#FFD700',
        Delay: '#666666'
    }
};

function CanvasOverlay({ subscribe, getDataRef, viewMode = 'live' }) {
    const canvasRef = useRef(null);

    // Double-buffering: store the LAST SUCCESSFULLY DECODED image
    const decodedImageRef = useRef(null);
    const pendingImageSrcRef = useRef(null);
    const isDecodingRef = useRef(false);

    const animationRef = useRef(null);
    const lastDataRef = useRef(null);

    // Async image decoder - loads image without blocking canvas
    const decodeImage = useCallback((base64) => {
        // Skip if already decoding or same image
        if (isDecodingRef.current || pendingImageSrcRef.current === base64) {
            return;
        }

        pendingImageSrcRef.current = base64;
        isDecodingRef.current = true;

        const img = new Image();
        img.onload = () => {
            decodedImageRef.current = img;
            isDecodingRef.current = false;
        };
        img.onerror = () => {
            isDecodingRef.current = false;
        };
        img.src = `data:image/jpeg;base64,${base64}`;
    }, []);

    // Draw board zone
    const drawBoardZone = useCallback((ctx, boardZone, showFill = true) => {
        if (!boardZone) return;
        const { x1, y1, x2, y2 } = boardZone;

        if (showFill) {
            ctx.fillStyle = COLORS.board.fill;
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }

        ctx.strokeStyle = COLORS.board.stroke;
        ctx.lineWidth = 2;
        ctx.shadowColor = COLORS.board.stroke;
        ctx.shadowBlur = 10;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0;

        // Corner brackets
        const bracketLen = 20;
        ctx.lineWidth = 3;
        const corners = [
            [[x1, y1 + bracketLen], [x1, y1], [x1 + bracketLen, y1]],
            [[x2 - bracketLen, y1], [x2, y1], [x2, y1 + bracketLen]],
            [[x1, y2 - bracketLen], [x1, y2], [x1 + bracketLen, y2]],
            [[x2 - bracketLen, y2], [x2, y2], [x2, y2 - bracketLen]]
        ];
        corners.forEach(corner => {
            ctx.beginPath();
            ctx.moveTo(corner[0][0], corner[0][1]);
            ctx.lineTo(corner[1][0], corner[1][1]);
            ctx.lineTo(corner[2][0], corner[2][1]);
            ctx.stroke();
        });

        // Zone label
        ctx.fillStyle = COLORS.board.stroke;
        ctx.font = '600 14px Inter, Roboto, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('BOARD ZONE', x1 + 5, y1 - 8);
    }, []);

    // Draw hand skeleton
    const drawHand = useCallback((ctx, handLabel, handData) => {
        if (!handData.visible || handData.landmarks.length === 0) return;

        const colors = handLabel === 'Left' ? COLORS.leftHand : COLORS.rightHand;
        const landmarks = handData.landmarks;

        // Scale landmarks
        const scaledLandmarks = landmarks.map(([x, y]) => [
            x * CANVAS_WIDTH,
            y * CANVAS_HEIGHT
        ]);

        // Draw connections
        ctx.strokeStyle = colors.main;
        ctx.lineWidth = 2;
        ctx.shadowColor = colors.main;
        ctx.shadowBlur = 8;

        for (const [start, end] of HAND_CONNECTIONS) {
            if (scaledLandmarks[start] && scaledLandmarks[end]) {
                ctx.beginPath();
                ctx.moveTo(scaledLandmarks[start][0], scaledLandmarks[start][1]);
                ctx.lineTo(scaledLandmarks[end][0], scaledLandmarks[end][1]);
                ctx.stroke();
            }
        }

        // Draw landmarks
        ctx.shadowBlur = 10;
        for (let i = 0; i < scaledLandmarks.length; i++) {
            const [x, y] = scaledLandmarks[i];
            const radius = [0, 4, 8, 12, 16, 20].includes(i) ? 6 : 4;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = colors.main;
            ctx.fill();
        }
        ctx.shadowBlur = 0;

        // Hand label and state
        const [cx, cy] = handData.center;
        const stateColor = COLORS.states[handData.state] || '#ffffff';

        ctx.fillStyle = stateColor;
        ctx.font = '600 16px Inter, Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.shadowColor = stateColor;
        ctx.shadowBlur = 10;
        ctx.fillText(`${handLabel[0]}: ${handData.state}`, cx, cy - 40);
        ctx.shadowBlur = 0;

        // Velocity
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Inter, Roboto, sans-serif';
        ctx.fillText(`${handData.velocity.toFixed(1)} px/f`, cx, cy - 25);
    }, []);

    // Helper: draw grid
    const drawGrid = useCallback((ctx, color = 'rgba(30, 144, 255, 0.05)') => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        for (let x = 0; x < CANVAS_WIDTH; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, CANVAS_HEIGHT);
            ctx.stroke();
        }
        for (let y = 0; y < CANVAS_HEIGHT; y += 50) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(CANVAS_WIDTH, y);
            ctx.stroke();
        }
    }, []);

    // Main draw function - NEVER waits for image decode
    const draw = useCallback((data) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        if (viewMode === 'live') {
            // Trigger async decode for new image
            if (data?.image) {
                decodeImage(data.image);
            }

            // ALWAYS draw the last successfully decoded image (no flicker!)
            if (decodedImageRef.current) {
                ctx.drawImage(decodedImageRef.current, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            } else {
                // No image yet - show placeholder
                ctx.fillStyle = '#0d1b2a';
                ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
                drawGrid(ctx);
            }
        } else if (viewMode === 'twin') {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            drawGrid(ctx, 'rgba(0, 255, 136, 0.08)');
        }

        if (!data) {
            ctx.fillStyle = '#1e90ff';
            ctx.font = '500 24px Inter, Roboto, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for backend connection...', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
            return;
        }

        // Draw board zone
        if (data.board_zone) {
            drawBoardZone(ctx, data.board_zone, viewMode === 'live');
        }

        // Draw hands
        for (const [handLabel, handData] of Object.entries(data.hands)) {
            drawHand(ctx, handLabel, handData);
        }

        // Frame info (only in live mode)
        if (viewMode === 'live' && data.frame_id !== undefined) {
            ctx.fillStyle = 'rgba(13, 27, 42, 0.85)';
            ctx.fillRect(10, CANVAS_HEIGHT - 35, 220, 25);
            ctx.fillStyle = '#1e90ff';
            ctx.font = '12px Inter, Roboto, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`Frame: ${data.frame_id} | Time: ${data.video_time?.toFixed(2) || 0}s`, 15, CANVAS_HEIGHT - 18);
        }
    }, [viewMode, decodeImage, drawGrid, drawBoardZone, drawHand]);

    // Animation loop - runs continuously at 60fps
    useEffect(() => {
        let isRunning = true;

        const animate = () => {
            if (!isRunning) return;
            const dataRef = getDataRef();
            // Always redraw to pick up newly decoded images
            draw(dataRef.current);
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

    return (
        <div className="canvas-container w-full h-full flex items-center justify-center">
            <canvas
                ref={canvasRef}
                width={CANVAS_WIDTH}
                height={CANVAS_HEIGHT}
                className="max-w-full max-h-full rounded-lg"
                style={{
                    aspectRatio: `${CANVAS_WIDTH}/${CANVAS_HEIGHT}`,
                    imageRendering: 'crisp-edges'
                }}
            />
        </div>
    );
}

export default CanvasOverlay;
