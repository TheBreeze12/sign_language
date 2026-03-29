import { useEffect, useMemo, useRef, useState, type ComponentProps } from 'react';
import { fetchSignPlaybackAsset, type SignPlaybackAsset } from '../../services/signPlayback';
import '../../styles/search.css';

const HOT_WORDS = ['LIBRARY', 'CANDY', 'PJS', 'DRY', 'THESE', 'TURN', 'ANY', 'DEEP'];
const UNITY_IFRAME_ORIGIN = '*';

type UnityBridgeMessage =
  | { type: 'set-source-path'; payload: { jsonPath: string; word: string } }
  | { type: 'set-playback-time'; payload: { currentTimeMs: number } }
  | { type: 'set-playback-state'; payload: { playing: boolean } }
  | { type: 'set-playback-rate'; payload: { playbackRate: number } };

function Search() {
  const [keyword, setKeyword] = useState('');
  const [selectedWord, setSelectedWord] = useState('LIBRARY');
  const [asset, setAsset] = useState<SignPlaybackAsset | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUnityReady, setIsUnityReady] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastSyncedTimeMsRef = useRef<number | null>(null);

  const handleSubmit: NonNullable<ComponentProps<'form'>['onSubmit']> = (event) => {
    event.preventDefault();
    const nextWord = keyword.trim();

    if (!nextWord) {
      return;
    }

    setSelectedWord(nextWord.toUpperCase());
  };

  const handleQuickSelect = (word: string) => {
    setKeyword(word);
    setSelectedWord(word);
  };

  const unitySceneUrl = useMemo(() => {
    if (!asset?.unityScenePath) {
      return '';
    }

    const sceneUrl = new URL(asset.unityScenePath, window.location.origin);
    sceneUrl.searchParams.set('word', selectedWord);
    return sceneUrl.toString();
  }, [asset?.unityScenePath, selectedWord]);

  const postToUnity = (message: UnityBridgeMessage) => {
    iframeRef.current?.contentWindow?.postMessage(message, UNITY_IFRAME_ORIGIN);
  };

  useEffect(() => {
    let ignore = false;

    const loadAsset = async () => {
      setIsLoading(true);

      try {
        const nextAsset = await fetchSignPlaybackAsset(selectedWord);
        if (!ignore) {
          setAsset(nextAsset);
        }
      } catch (error) {
        if (!ignore) {
          setAsset(null);
        }
      } finally {
        if (!ignore) {
          setIsLoading(false);
        }
      }
    };

    void loadAsset();

    return () => {
      ignore = true;
    };
  }, [selectedWord]);

  const sendSourcePathToUnity = (jsonPath: string, word: string) => {
    postToUnity({
      type: 'set-source-path',
      payload: { jsonPath, word },
    });
  };

  const syncUnityPlaybackTime = (currentTimeSeconds: number) => {
    const currentTimeMs = Math.round(currentTimeSeconds * 1000);
    if (lastSyncedTimeMsRef.current === currentTimeMs) {
      return;
    }

    lastSyncedTimeMsRef.current = currentTimeMs;
    postToUnity({
      type: 'set-playback-time',
      payload: {
        currentTimeMs,
      },
    });
  };

  const handleVideoLoadedMetadata: NonNullable<ComponentProps<'video'>['onLoadedMetadata']> = (
    event,
  ) => {
    syncUnityPlaybackTime(event.currentTarget.currentTime);
  };

  const handleVideoSeeked: NonNullable<ComponentProps<'video'>['onSeeked']> = (event) => {
    syncUnityPlaybackTime(event.currentTarget.currentTime);
  };

  const handleVideoPlayState = (playing: boolean) => {
    postToUnity({
      type: 'set-playback-state',
      payload: { playing },
    });
  };

  const syncUnityPlaybackRate = (playbackRate: number) => {
    postToUnity({
      type: 'set-playback-rate',
      payload: { playbackRate },
    });
  };

  const handleVideoPlay: NonNullable<ComponentProps<'video'>['onPlay']> = (event) => {
    const video = event.currentTarget;
    syncUnityPlaybackTime(video.currentTime);
    handleVideoPlayState(true);
    window.requestAnimationFrame(() => {
      syncUnityPlaybackTime(video.currentTime);
    });
    window.setTimeout(() => {
      syncUnityPlaybackTime(video.currentTime);
    }, 80);
  };

  const handleVideoPause: NonNullable<ComponentProps<'video'>['onPause']> = (event) => {
    syncUnityPlaybackTime(event.currentTarget.currentTime);
    handleVideoPlayState(false);
  };

  const handleVideoEnded: NonNullable<ComponentProps<'video'>['onEnded']> = (event) => {
    syncUnityPlaybackTime(event.currentTarget.currentTime);
    handleVideoPlayState(false);
  };

  const handleVideoRateChange: NonNullable<ComponentProps<'video'>['onRateChange']> = (event) => {
    syncUnityPlaybackRate(event.currentTarget.playbackRate);
  };

  useEffect(() => {
    const handleUnityMessage = (event: MessageEvent) => {
      if (event.data?.type === 'unity-ready') {
        setIsUnityReady(true);
      }
    };

    window.addEventListener('message', handleUnityMessage);
    return () => {
      window.removeEventListener('message', handleUnityMessage);
    };
  }, []);

  useEffect(() => {
    setIsUnityReady(false);
  }, [unitySceneUrl]);

  useEffect(() => {
    if (!isUnityReady || !asset?.jsonPath) {
      return;
    }

    sendSourcePathToUnity(asset.jsonPath, selectedWord);

    const video = videoRef.current;
    if (!video) {
      return;
    }

    lastSyncedTimeMsRef.current = null;
    syncUnityPlaybackTime(video.currentTime);
    syncUnityPlaybackRate(video.playbackRate);
    handleVideoPlayState(!video.paused && !video.ended);
  }, [asset?.jsonPath, isUnityReady, selectedWord]);

  return (
    <div className="search-page">
      <section className="search-shell">
        <div className="search-shell__glow search-shell__glow--left" />
        <div className="search-shell__glow search-shell__glow--right" />

        <div className="search-stage">
          <div className="search-column search-column--left">
            <form className="search-form" onSubmit={handleSubmit}>
              <label className="search-form__field" htmlFor="sign-search-input">
                <div className="search-form__control">
                  <input
                    id="sign-search-input"
                    type="text"
                    value={keyword}
                    onChange={(event) => setKeyword(event.target.value)}
                    placeholder="Try HELLO / THANK YOU / FAMILY"
                  />
                  <button type="submit">Search</button>
                </div>
              </label>
            </form>

            <div className="search-hotwords" aria-label="Popular sign words">
              {HOT_WORDS.map((word) => (
                <button
                  key={word}
                  type="button"
                  className={`search-hotwords__chip ${selectedWord === word ? 'is-active' : ''}`}
                  onClick={() => handleQuickSelect(word)}
                >
                  {word}
                </button>
              ))}
              <div className="search-hotwords__hint" aria-label="Unity scene tips">
                <button type="button" className="search-hotwords__hint-icon" aria-label="查看 Unity Scene 操作提示">
                  ?
                </button>
                <div className="search-hotwords__tooltip" role="tooltip">
                  <p className="search-hotwords__tooltip-line--single">加载 Unity Scene 需要一定时间，请稍等。</p>
                  <p className="search-hotwords__tooltip-line--single">摁住 Alt + 左键可以旋转视角，摁住中键可以平移视角。</p>
                </div>
              </div>
            </div>
            <section className="search-panel search-panel--model">
              <div className="unity-preview">
                <div className="unity-preview__frame">
                  <span className="search-card-tag">Unity Scene</span>
                  <div className="unity-preview__window">
                    <div className="unity-preview__window-bar" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </div>

                    {unitySceneUrl ? (
                      <iframe
                        ref={iframeRef}
                        className="unity-preview__iframe"
                        title="Unity WebGL Scene"
                        src={unitySceneUrl}
                        allow="autoplay; fullscreen"
                      />
                    ) : (
                      <div className="unity-preview__placeholder">
                        <div className="unity-preview__badge">WebGL Ready</div>
                        <h3>{isLoading ? '加载中' : 'Unity Scene 提示'}</h3>
                        <p>加载 Unity Scene 需要一定时间，请稍等。</p>
                        <p>摁住 Alt + 左键可以旋转视角，摁住中键可以平移视角。</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          </div>

          <section className="search-panel search-panel--video">
            <div className="video-preview">
              <div className="video-preview__screen">
                <span className="search-card-tag">Video</span>
                <div className="video-preview__overlay" />
                <div className="video-preview__content">
                  {asset?.videoPath ? (
                    <video
                      ref={videoRef}
                      className="video-preview__player"
                      src={asset.videoPath}
                      controls
                      preload="metadata"
                      onLoadedMetadata={handleVideoLoadedMetadata}
                      onSeeked={handleVideoSeeked}
                      onPlay={handleVideoPlay}
                      onPause={handleVideoPause}
                      onEnded={handleVideoEnded}
                      onRateChange={handleVideoRateChange}
                    />
                  ) : (
                    <div className="video-preview__empty">
                      <div className="video-preview__play" aria-hidden="true" />
                      <p>{isLoading ? 'Loading video path...' : 'Waiting for video path'}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}

export default Search;
