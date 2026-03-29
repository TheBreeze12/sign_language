using System;

[Serializable]
public class GestureFrameData
{
    public int frame_index;
    public long timestamp_ms;
    public int hand_count;
    public HandData[] hands;

    public GestureData ToGestureData()
    {
        return new GestureData
        {
            hand_count = hand_count,
            hands = hands ?? Array.Empty<HandData>()
        };
    }
}

[Serializable]
public class GestureFrameCollection
{
    public GestureFrameData[] frames;
}
