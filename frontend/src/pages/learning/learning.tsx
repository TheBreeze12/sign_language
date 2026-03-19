import Banana from '../../assets/image/banana.png'
import Hello from '../../assets/image/hello.png'
import Baby from '../../assets/image/baby.png'
import Bread from '../../assets/image/bread.png'
import Halloween from '../../assets/image/Halloween.png'

import DynamicWordBox from '../../shared/components/DynamicWordBOX.tsx';

import '../../styles/learning.css'
import {useRef} from "react";

export default function Learning() {
    const sliderRef = useRef<HTMLDivElement>(null);

    // 下一张的逻辑（保持你原有的原生 DOM 操作）
    const handleNext = () => {
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children;
            if (items.length > 0) {
                slider.appendChild(items[0]);
            }
        }
    }

    // 上一张的逻辑
    const handlePrev = () => {
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children;
            if (items.length > 0) {
                slider.prepend(items[items.length - 1]);
            }
        }
    }

    // 把你要展示的单词和封面图配置成一个数组，方便循环渲染
    const vocabularyList = [
        { word: 'BANANA', img: Banana },
        { word: 'HELLO', img: Hello },
        { word: 'BABY', img: Baby },
        { word: 'BREAD', img: Bread },
        { word: 'HALLOWEEN', img: Halloween },
    ];

    return (
        <div className='container'>
            {/* 轮播图主体 */}
            <div className='slider' ref={sliderRef}>
                {vocabularyList.map((item) => (
                    // 核心关联点：在这里调用子组件，把 word 和 img 传过去
                    <DynamicWordBox
                        key={item.word}
                        word={item.word}
                        previewImg={item.img}
                    />
                ))}
            </div>

            {/* 左右切换按钮 */}
            <div className='buttons'>
                <span className='prev' onClick={handlePrev}>&lt;</span>
                <span className='next' onClick={handleNext}>&gt;</span>
            </div>
        </div>
    )
}