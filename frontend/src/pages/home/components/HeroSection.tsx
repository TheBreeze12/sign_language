import React from 'react';
// 1. 引入你的自定义跳转 Hook（请根据你实际的文件夹层级调整相对路径）
import { useBlinkNavigate } from '../../../shared/components/blinktransition.tsx';

const HeroSection: React.FC = () => {
    // 2. 初始化 Hook
    const blinkNavigate = useBlinkNavigate();

    // 3. 定义点击事件
    const handleExploreClick = () => {
        blinkNavigate('/translate');
    };

    return (
        <div className="content">

            <div className="titleMain">
                <svg width="430" height="200" viewBox="0 0 430 116" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {/* SVG Path 省略，保持原样即可 */}
                    <path d="M73.088 52.608C72.4907 53.12... (省略路径代码) ...412.354 32 410.477 32Z" fill="white"/>
                </svg>
            </div>

            <div className="info-wrap">
                <p>
                    ASL eye is a website as a Real-time Sign Language Translator,for those who also hope communicating with the world.
                </p>
            </div>

            <div className="cta">
                {/* 4. 在按钮上绑定 onClick 事件 */}
                <button onClick={handleExploreClick}>
                    Explore More
                    <i className="fa-solid fa-arrow-right"></i>
                </button>
            </div>

            {/*<div className="slider">
                <i className="fa-solid fa-chevron-left"></i>
                <i className="fa-solid fa-chevron-right"></i>
            </div>*/}
        </div>
    );
};

export default HeroSection;