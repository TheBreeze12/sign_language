import { useState, useEffect } from "react";

export interface WordBoxProps {
    word: string;       // 词汇，如 "BANANA"
    previewImg: string; // 封面图的本地路径
}

export default function DynamicWordBox({ word, previewImg }: WordBoxProps) {
    // 因为明确是视频，我们只需要存一个视频的 URL
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchVideo = async () => {
            try {
                const response = await fetch(`/api/sign/downloads?name=${word.toUpperCase()}`);
                const result = await response.json();

                // 假设后端返回的数据格式是 { code: 200, data: ["/path/to/video.mp4", ...] }
                // 我们取数组的第一个作为要播放的视频
                if (result.code === 200 && result.data && result.data.length > 0) {
                    setVideoUrl(result.data[0]);
                }
            } catch (error) {
                console.error(`Failed to load video for ${word}`, error);
            } finally {
                setLoading(false);
            }
        };

        fetchVideo();
    }, [word]);

    return (
        <div className='box'>
            <img src={previewImg} alt={word} className='preview'/>

            <div className='clipBox'>
                {/* 1. 加载中状态 */}
                {loading ? (
                        <div style={{ color: 'white', padding: '20px', textAlign: 'center' }}>
                            Loading...
                        </div>
                    ) :

                    /* 2. 成功拿到视频 URL：直接渲染 video 标签 */
                    videoUrl ? (
                            <video src={videoUrl} muted loop playsInline autoPlay />
                        ) :

                        /* 3. 兜底：接口没报错但没查到数据 */
                        (
                            <div style={{ color: '#ff4444', padding: '20px', textAlign: 'center' }}>
                                No video found
                            </div>
                        )}
            </div>
        </div>
    );
}