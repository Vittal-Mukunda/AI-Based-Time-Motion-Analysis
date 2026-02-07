import React, { useState } from 'react';
import useWebSocket from './hooks/useWebSocket';
import { DashboardProvider } from './context/DashboardContext';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import TabManager from './components/TabManager';
import CanvasOverlay from './components/CanvasOverlay';
import DigitalTwinView from './components/DigitalTwinView';
import AnalyticsCommandCenter from './components/AnalyticsCommandCenter';

function AppContent({ isConnected, error, subscribe, getDataRef, activeTab, setActiveTab }) {
    const dataRef = getDataRef();

    const renderView = () => {
        switch (activeTab) {
            case 'live':
                return (
                    <CanvasOverlay
                        subscribe={subscribe}
                        getDataRef={getDataRef}
                        viewMode="live"
                    />
                );
            case 'twin':
                return (
                    <DigitalTwinView
                        subscribe={subscribe}
                        getDataRef={getDataRef}
                    />
                );
            case 'analytics':
                return <AnalyticsCommandCenter />;
            default:
                return null;
        }
    };

    return (
        <div className="h-screen w-screen flex flex-col bg-exec-bg overflow-hidden">
            {/* Header */}
            <Header
                isConnected={isConnected}
                fps={dataRef.current?.fps}
                frameId={dataRef.current?.frame_id}
            />

            {/* Tab Navigation */}
            <TabManager activeTab={activeTab} onTabChange={setActiveTab} />

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Canvas/View Area */}
                <main className={`flex-1 p-4 flex items-center justify-center ${activeTab === 'analytics' ? '' : ''}`}>
                    <div className={`w-full h-full view-transition ${activeTab === 'analytics' ? '' : 'max-w-5xl'}`}>
                        {renderView()}
                    </div>
                </main>

                {/* Sidebar - only show on live view */}
                {activeTab === 'live' && (
                    <Sidebar dataRef={dataRef} subscribe={subscribe} />
                )}
            </div>

            {/* Error Toast */}
            {error && (
                <div className="fixed bottom-4 left-4 bg-red-900/90 border border-red-500 text-red-200 px-4 py-2 rounded-lg font-medium">
                    {error}
                </div>
            )}
        </div>
    );
}

function App() {
    const { isConnected, error, subscribe, getDataRef } = useWebSocket();
    const [activeTab, setActiveTab] = useState('live');

    return (
        <DashboardProvider subscribe={subscribe} getDataRef={getDataRef}>
            <AppContent
                isConnected={isConnected}
                error={error}
                subscribe={subscribe}
                getDataRef={getDataRef}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
            />
        </DashboardProvider>
    );
}

export default App;
