using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class GestureSyncController : MonoBehaviour
{
    [SerializeField] private Test test;
    [SerializeField] private bool autoLoadOnStart = false;
    [SerializeField] private string initialGesturePath = "";
    [SerializeField] private bool autoPlayLoadedGesture = true;
    [SerializeField] private bool loopPlayback = true;

    private readonly List<GestureFrameData> frames = new List<GestureFrameData>();
    private long currentTimeMs;
    private bool isPlaying;
    private float playbackRate = 1f;
    private float lastRealtimeSinceStartup;
    private Coroutine loadGestureCoroutine;

    private void Start()
    {
        if (autoLoadOnStart && !string.IsNullOrWhiteSpace(initialGesturePath))
        {
            SetGestureSourcePath(initialGesturePath);
        }

        lastRealtimeSinceStartup = Time.realtimeSinceStartup;
    }

    private void Update()
    {
        if (!isPlaying || frames.Count == 0)
        {
            lastRealtimeSinceStartup = Time.realtimeSinceStartup;
            return;
        }

        float now = Time.realtimeSinceStartup;
        float deltaSeconds = now - lastRealtimeSinceStartup;
        lastRealtimeSinceStartup = now;

        if (deltaSeconds <= 0f)
        {
            return;
        }

        long nextTimeMs = currentTimeMs + (long)(deltaSeconds * 1000f * playbackRate);
        long maxTimeMs = frames[frames.Count - 1].timestamp_ms;

        if (nextTimeMs > maxTimeMs)
        {
            if (loopPlayback)
            {
                nextTimeMs = 0;
            }
            else
            {
                nextTimeMs = maxTimeMs;
                isPlaying = false;
            }
        }

        currentTimeMs = nextTimeMs;
        ApplyFrameForTime(currentTimeMs);
    }

    public void SetGestureSourcePath(string sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            Debug.LogWarning("Gesture source path is empty.");
            return;
        }

        if (loadGestureCoroutine != null)
        {
            StopCoroutine(loadGestureCoroutine);
            loadGestureCoroutine = null;
        }

        if (IsWebUrl(sourcePath))
        {
            loadGestureCoroutine = StartCoroutine(LoadGestureFromUrl(sourcePath));
            return;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        Debug.LogWarning("WebGL does not support loading gesture data from a local file path: " + sourcePath);
        return;
#endif

        if (!File.Exists(sourcePath))
        {
            Debug.LogWarning("Gesture source path not found: " + sourcePath);
            return;
        }

        SetGestureRawText(File.ReadAllText(sourcePath));
    }

    private static bool IsWebUrl(string sourcePath)
    {
        return sourcePath.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
               sourcePath.StartsWith("https://", StringComparison.OrdinalIgnoreCase);
    }

    private System.Collections.IEnumerator LoadGestureFromUrl(string sourceUrl)
    {
        using (UnityWebRequest request = UnityWebRequest.Get(sourceUrl))
        {
            yield return request.SendWebRequest();

            loadGestureCoroutine = null;

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("Failed to load gesture source from URL: " + sourceUrl + ". " + request.error);
                yield break;
            }

            SetGestureRawText(request.downloadHandler.text);
        }
    }

    public void SetGestureRawText(string rawText)
    {
        frames.Clear();

        if (string.IsNullOrWhiteSpace(rawText))
        {
            Debug.LogWarning("Gesture raw text is empty.");
            return;
        }

        string trimmed = rawText.Trim();
        if (trimmed.StartsWith("{", StringComparison.Ordinal) &&
            trimmed.Contains("\"frames\"", StringComparison.Ordinal))
        {
            try
            {
                GestureFrameCollection collection = JsonUtility.FromJson<GestureFrameCollection>(trimmed);
                if (collection?.frames != null)
                {
                    frames.AddRange(collection.frames);
                }
            }
            catch (ArgumentException exception)
            {
                Debug.LogWarning("Failed to parse frame collection JSON, falling back to JSONL parsing. " + exception.Message);
            }
        }

        if (frames.Count == 0)
        {
            string[] lines = rawText.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);
            foreach (string line in lines)
            {
                string jsonLine = line.Trim();
                if (string.IsNullOrWhiteSpace(jsonLine))
                {
                    continue;
                }

                try
                {
                    GestureFrameData frame = JsonUtility.FromJson<GestureFrameData>(jsonLine);
                    if (frame != null)
                    {
                        frames.Add(frame);
                    }
                }
                catch (ArgumentException exception)
                {
                    Debug.LogWarning("Skipped invalid gesture frame JSON: " + exception.Message);
                }
            }
        }

        frames.Sort((left, right) => left.timestamp_ms.CompareTo(right.timestamp_ms));
        currentTimeMs = 0;
        ApplyFrameForTime(currentTimeMs);

        if (autoPlayLoadedGesture && frames.Count > 0)
        {
            isPlaying = true;
            lastRealtimeSinceStartup = Time.realtimeSinceStartup;
        }
    }

    public void SetPlaybackTimeMs(string playbackTimeMs)
    {
        if (!long.TryParse(playbackTimeMs, out long parsed))
        {
            Debug.LogWarning("Invalid playback time: " + playbackTimeMs);
            return;
        }

        currentTimeMs = Math.Max(0, parsed);
        ApplyFrameForTime(currentTimeMs);
    }

    public void SetPlaybackState(string playing)
    {
        if (!bool.TryParse(playing, out bool parsed))
        {
            Debug.LogWarning("Invalid playback state: " + playing);
            return;
        }

        isPlaying = parsed;
        lastRealtimeSinceStartup = Time.realtimeSinceStartup;
    }

    public void SetPlaybackRate(string rate)
    {
        if (!float.TryParse(rate, out float parsed) || parsed <= 0f)
        {
            Debug.LogWarning("Invalid playback rate: " + rate);
            return;
        }

        playbackRate = parsed;
        lastRealtimeSinceStartup = Time.realtimeSinceStartup;
    }

    private void ApplyFrameForTime(long playbackTimeMs)
    {
        if (test == null || frames.Count == 0)
        {
            return;
        }

        int frameIndex = FindFrameIndex(playbackTimeMs);

        if (frameIndex < 0 || frameIndex >= frames.Count)
        {
            return;
        }

        test.ApplyGestureData(frames[frameIndex].ToGestureData());
    }

    private int FindFrameIndex(long playbackTimeMs)
    {
        int low = 0;
        int high = frames.Count - 1;
        int result = 0;

        while (low <= high)
        {
            int mid = (low + high) / 2;
            if (frames[mid].timestamp_ms <= playbackTimeMs)
            {
                result = mid;
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }

        return result;
    }
}
