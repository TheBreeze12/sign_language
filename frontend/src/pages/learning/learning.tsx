import { useRef } from "react";
import Banana from '../../assets/image/banana.png'
import Hello from '../../assets/image/hello.png'
import Baby from '../../assets/image/baby.png'
import Bread from '../../assets/image/bread.png'
import Halloween from '../../assets/image/Halloween.png'

import DynamicWordBox from '../../shared/components/DynamicWordBOX.tsx';
import '../../styles/learning.css'

export default function Learning() {
    const sliderRef = useRef<HTMLDivElement>(null);

    const handleNext = () => {
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children;
            if (items.length > 0) {
                slider.appendChild(items[0]);
            }
        }
    }

    const handlePrev = () => {
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children;
            if (items.length > 0) {
                slider.prepend(items[items.length - 1]);
            }
        }
    }

    // 新增：把对应视频的 vidref 添加到字典中
    const vocabularyList = [
        {
            word: 'BANANA',
            img: Banana,
            // 专业的 ASL Banana 演示
            videoUrl: 'https://media.giphy.com/media/3o6Zt6tOaFxXVjTxg4/giphy.mp4'
        },
        {
            word: 'HELLO',
            img: Hello,
            // Sign with Robert: Hello
            videoUrl: 'https://media.giphy.com/media/3o7TKNKOfKlIhbD3gY/giphy.mp4'
        },
        {
            word: 'BABY',
            img: Baby,
            // Sign with Robert: Baby
            videoUrl: 'https://media.giphy.com/media/l0HlCuPLK3fVM7urm/giphy.mp4'
        },
        {
            word: 'BREAD',
            img: Bread,
            // Sign with Robert: Bread
            videoUrl: 'https://media.giphy.com/media/l0MYQ98VjXVUzm2Mo/giphy.mp4'
        },
        {
            word: 'HALLOWEEN',
            img: Halloween,
            // Sign with Robert: Halloween
            videoUrl: 'https://media.giphy.com/media/l0MYN09eUAzRiE1kA/giphy.mp4'
        },
    ];

    return (
        <div className='container'>
            <div className='slider' ref={sliderRef}>
                {vocabularyList.map((item) => (
                    <DynamicWordBox
                        key={item.word}
                        word={item.word}
                        previewImg={item.img}
                        videoUrl={item.videoUrl}
                    />
                ))}
            </div>

            <div className='buttons'>
                <span className='prev' onClick={handlePrev}>&lt;</span>
                <span className='next' onClick={handleNext}>&gt;</span>
            </div>
        </div>
    )
}