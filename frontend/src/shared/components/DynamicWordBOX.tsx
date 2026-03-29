
export interface WordBoxProps {
    word: string;
    previewImg: string;
    videoUrl: string;
}

export default function DynamicWordBox({ word, previewImg, videoUrl }: WordBoxProps) {
    return (
        <div className='box'>
            <img src={previewImg} alt={word} className='preview'/>

            <div className='clipBox'>
                <video
                    src={videoUrl}
                    muted
                    loop
                    playsInline
                    autoPlay
                    style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        borderRadius: '15px'
                    }}
                />
            </div>
        </div>
    );
}