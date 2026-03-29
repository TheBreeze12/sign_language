export type SignPlaybackAsset = {
  word: string;
  videoPath: string;
  jsonPath: string;
  unityScenePath: string;
};

type SignResourceItem = {
  stem: string;
  word: string;
  json_url: string;
  video_url: string;
};

type SignResourcesResponse = {
  code: number;
  msg?: string;
  data?: {
    word: string;
    count: number;
    items: SignResourceItem[];
  };
};

const API_BASE_URL = 'http://127.0.0.1:5000';
const UNITY_SCENE_PATH = 'http://127.0.0.1:18081';


export async function fetchSignPlaybackAsset(word: string): Promise<SignPlaybackAsset> {
  const normalizedWord = word.trim().toUpperCase();

  if (!normalizedWord) {
    throw new Error('Word is required.');
  }

  const response = await fetch(
    `${API_BASE_URL}/api/sign/resources?name=${encodeURIComponent(normalizedWord)}`,
  );

  const payload = (await response.json()) as SignResourcesResponse;

  if (!response.ok || payload.code !== 200 || !payload.data?.items?.length) {
    throw new Error(payload.msg || 'Resource not found.');
  }

  const primaryItem = payload.data.items[0];


  return {
    word: payload.data.word,
    videoPath: primaryItem.video_url,
    jsonPath: primaryItem.json_url,
    unityScenePath: UNITY_SCENE_PATH,
  };
}
