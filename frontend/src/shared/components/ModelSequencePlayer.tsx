import React, { useRef, Suspense, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF, Html, useProgress } from '@react-three/drei';
import * as THREE from 'three';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

const Loader = () => {
    const { progress } = useProgress();
    return (
        <Html center>
            <div style={{ color: '#00f0ff', whiteSpace: 'nowrap', fontWeight: 'bold' }}>
                Loading Models... {progress.toFixed(0)}%
            </div>
        </Html>
    );
};

interface SequenceProps {
    urls: string[];
    fps?: number;
    playTrigger: number;         // 1. 新增：接收外部的重播信号
    onFinished: () => void;      // 2. 新增：播放结束的回调
}

const SequenceController: React.FC<SequenceProps> = ({ urls, fps = 12, playTrigger, onFinished }) => {
    // @ts-ignore
    const gltfs = useGLTF(urls) as any[];

    const groupRef = useRef<THREE.Group>(null);
    const clockRef = useRef(0);
    const currentIndexRef = useRef(0);
    const isFinishedRef = useRef(false);

    // 当 playTrigger 改变（用户点击重播）或模型更新时，重置播放状态
    useEffect(() => {
        if (groupRef.current && gltfs.length > 0) {
            currentIndexRef.current = 0;
            clockRef.current = 0;
            isFinishedRef.current = false;

            // 隐藏所有帧，仅显示第一帧
            groupRef.current.children.forEach((child, index) => {
                child.visible = index === 0;
            });
        }
    }, [playTrigger, gltfs]);

    // 高性能帧渲染循环
    useFrame((_, delta) => {
        if (!groupRef.current || gltfs.length === 0 || isFinishedRef.current) return;

        clockRef.current += delta;

        // 当经过的时间大于一帧所需时间时
        if (clockRef.current > 1 / fps) {
            // 如果已经到了最后一帧
            if (currentIndexRef.current >= gltfs.length - 1) {
                isFinishedRef.current = true;
                onFinished(); // 告诉父组件播放完毕，显示重播按钮
                return;
            }

            const children = groupRef.current.children;

            // 隐藏当前帧
            if (children[currentIndexRef.current]) {
                children[currentIndexRef.current].visible = false;
            }

            // 索引 +1，不再循环
            currentIndexRef.current++;

            // 显示下一帧
            if (children[currentIndexRef.current]) {
                children[currentIndexRef.current].visible = true;
            }

            // 减去时间以保持节奏精准
            clockRef.current -= 1 / fps;
        }
    });

    return (
        <group ref={groupRef}>
            {gltfs.map((gltf, index) => (
                <primitive key={urls[index]} object={gltf.scene} visible={index === 0} />
            ))}
        </group>
    );
};

interface ModelSequencePlayerProps {
    isStreaming: boolean;
    isProcessing: boolean;
    glbUrls: string[];
    fps?: number;
}

const ModelSequencePlayer: React.FC<ModelSequencePlayerProps> = ({
                                                                     isStreaming,
                                                                     isProcessing,
                                                                     glbUrls,
                                                                     fps = 5
                                                                 }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLDivElement>(null);

    // 控制重播的内部状态
    const [playTrigger, setPlayTrigger] = useState(0);
    const [isFinished, setIsFinished] = useState(false);

    // 当有了新模型时，预加载并自动重置播放状态，准备第一次播放
    useEffect(() => {
        if (glbUrls.length > 0) {
            useGLTF.preload(glbUrls);
            setIsFinished(false);
            setPlayTrigger(prev => prev + 1);
        }
    }, [glbUrls]);

    // 模型加载完毕时的淡入动画
    useGSAP(() => {
        if (glbUrls.length > 0 && canvasRef.current && !isStreaming) {
            gsap.fromTo(canvasRef.current,
                { opacity: 0, scale: 0.95 },
                { opacity: 1, scale: 1, duration: 1.2, ease: 'power2.out' }
            );
        }
    }, [glbUrls.length, isStreaming]);

    // 点击重播的处理函数
    const handleReplay = () => {
        setIsFinished(false);
        setPlayTrigger(prev => prev + 1);
    };

    return (
        <div className="ai-output-wrapper" ref={containerRef}>
            <div className="output-3d-container" style={{ position: 'relative' }}>

                {!isStreaming && !isProcessing && glbUrls.length === 0 && (
                    <div className="empty-state">Waiting for your sign language...</div>
                )}

                {isStreaming && glbUrls.length === 0 && (
                    <div className="empty-state recording-pulse" style={{ color: '#00f0ff'}}>
                        Recording in progress...
                    </div>
                )}

                {isProcessing && (
                    <div className="loading-state">Generating 3D Translation...</div>
                )}

                {/* 渲染模型 */}
                {glbUrls.length > 0 && (
                    <div className="canvas-wrapper" ref={canvasRef}>
                        <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
                            <ambientLight intensity={0.5} />
                            <directionalLight position={[10, 10, 5]} intensity={1} />

                            <Suspense fallback={<Loader />}>
                                <Stage environment="city" intensity={0.6}>
                                    <SequenceController
                                        urls={glbUrls}
                                        fps={fps}
                                        playTrigger={playTrigger}
                                        onFinished={() => setIsFinished(true)}
                                    />
                                </Stage>
                            </Suspense>

                            <OrbitControls autoRotate={false} />
                        </Canvas>

                        {/* 重播浮窗按钮（仅在播放完成时显示） */}
                        {isFinished && !isStreaming && (
                            <button
                                onClick={handleReplay}
                                style={{
                                    position: 'absolute',
                                    bottom: '30px',
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    zIndex: 10,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '10px 24px',
                                    backgroundColor: 'rgba(0, 240, 255, 0.1)',
                                    border: '1px solid #00f0ff',
                                    color: '#00f0ff',
                                    borderRadius: '30px',
                                    cursor: 'pointer',
                                    fontSize: '14px',
                                    fontWeight: 'bold',
                                    backdropFilter: 'blur(8px)',
                                    boxShadow: '0 4px 15px rgba(0, 240, 255, 0.2)',
                                    transition: 'all 0.3s ease'
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = 'rgba(0, 240, 255, 0.25)';
                                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 240, 255, 0.4)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = 'rgba(0, 240, 255, 0.1)';
                                    e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 240, 255, 0.2)';
                                }}
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                    <path d="M3 3v5h5" />
                                </svg>
                                REPLAY
                            </button>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
};

export default ModelSequencePlayer;