using System.Runtime.InteropServices;

/// <summary>
/// Class with JavaScript plugin functions for WebGL.
/// </summary>
public static class WebGLPlugin
{
    // Import "GetExternalJS" plugin function
    [DllImport("__Internal")]
    public static extern void GetExternalJS();
    // Import "SetTFJSBackend" plugin function
    [DllImport("__Internal")]
    public static extern void SetTFJSBackend(string backend);
    // Import "InitTFJSModel" plugin function
    [DllImport("__Internal")]
    public static extern void InitTFJSModel(string model_path, float[] mean, float[] std);
    // Import "UpdateOutputArray" plugin function
    [DllImport("__Internal")]
    public static extern void UpdateOutputArray(float[] output_data, int size);
    // Import "PerformInference" plugin function
    [DllImport("__Internal")]
    public static extern bool PerformInference(byte[] image_data, int size, int width, int height);
}