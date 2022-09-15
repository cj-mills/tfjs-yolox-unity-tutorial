using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Networking;
using System.Linq;


#if UNITY_EDITOR
using UnityEditor;

[InitializeOnLoad]
public class Startup
{
    [System.Serializable]
    class ModelData
    {
        public string name;
        public string path;

        public ModelData(string name, string path)
        {
            this.name = name;
            this.path = path;
        }
    }

    [System.Serializable]
    class ModelList
    {
        public List<ModelData> models;

        public ModelList(List<ModelData> models)
        {
            this.models = models;
        }
    }

    static Startup()
    {
        string tfjsModelsDir = "TFJSModels";
        List<ModelData> models = new List<ModelData>();

        Debug.Log("Available models");
        // Get the paths for each model folder
        foreach (string dir in Directory.GetDirectories($"{Application.streamingAssetsPath}/{tfjsModelsDir}"))
        {
            string dirStr = dir.Replace("\\", "/");
            // Extract the model folder name
            string[] splits = dirStr.Split('/');
            string modelName = splits[splits.Length - 1];

            // Get the paths for the model.json file for each model
            foreach (string file in Directory.GetFiles(dirStr))
            {
                if (file.EndsWith("model.json"))
                {
                    string fileStr = file.Replace("\\", "/").Replace(Application.streamingAssetsPath, "");
                    models.Add(new ModelData(modelName, fileStr));
                }
            }
        }
        ModelList modelList = new ModelList(models);
        string json = JsonUtility.ToJson(modelList);
        Debug.Log($"Model List JSON: {json}");
        using StreamWriter writer = new StreamWriter($"{Application.streamingAssetsPath}/models.json");
        writer.Write(json);
    }
}
#endif

public class YOLOXTFJS : MonoBehaviour
{
    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;
    [Tooltip("Mirror the in-game screen.")]
    public bool mirrorScreen = true;

    [Header("Data Processing")]
    [Tooltip("The target minimum model input dimensions")]
    public int targetDim = 224;

    [Header("Output Processing")]
    [Tooltip("A json file containing the colormaps for object classes")]
    public TextAsset colormapFile;
    [Tooltip("Minimum confidence score for keeping detected objects")]
    [Range(0, 1f)]
    public float minConfidence = 0.5f;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;

    [Header("Webcam")]
    [Tooltip("Use a webcam as input")]
    public bool useWebcam = false;
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);
    [Tooltip("The requested webcam framerate")]
    [Range(0, 60)]
    public int webcamFPS = 60;

    [Header("GUI")]
    [Tooltip("Display predicted class")]
    public bool displayBoundingBoxes = true;
    [Tooltip("Display number of detected objects")]
    public bool displayProposalCount = true;
    [Tooltip("Display fps")]
    public bool displayFPS = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0, 99)]
    public int fontScale = 50;
    [Tooltip("The number of seconds to wait between refreshing the fps value")]
    [Range(0.01f, 1.0f)]
    public float fpsRefreshRate = 0.1f;
    [Tooltip("The toggle for using a webcam as the input source")]
    public Toggle useWebcamToggle;
    [Tooltip("The dropdown menu that lists available webcam devices")]
    public Dropdown webcamDropdown;
    [Tooltip("The dropdown menu that lists available TFJS models")]
    public Dropdown modelDropdown;
    [Tooltip("The dropdown menu that lists available TFJS backends")]
    public Dropdown backendDropdown;

    [Header("TFJS")]
    [Tooltip("The name of the TFJS models folder")]
    public string tfjsModelsDir = "TFJSModels";

    // List of available webcam devices
    private WebCamDevice[] webcamDevices;
    // Live video input from a webcam
    private WebCamTexture webcamTexture;
    // The name of the current webcam  device
    private string currentWebcam;

    // The test image dimensions
    private Vector2Int imageDims;
    // The test image texture
    private Texture imageTexture;
    // The current screen object dimensions
    private Vector2Int screenDims;
    // The model GPU input texture
    private RenderTexture inputTextureGPU;
    // The model CPU input texture
    private Texture2D inputTextureCPU;

    // Stores the number of detected objects
    private int numObjects;

    // A class for parsing in colormaps from a JSON file
    [System.Serializable]
    class ColorMap { public string label; public float[] color; }
    // A class for reading in a list of colormaps from a JSON file
    [System.Serializable]
    class ColorMapList { public List<ColorMap> items; }
    // Stores a list of colormaps from a JSON file
    private ColorMapList colormapList;
    // A list of colors that map to class labels
    private Color[] colors;
    // A list of single pixel textures that map to class labels
    private Texture2D[] colorTextures;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;

    // File paths for the available TFJS models
    private List<string> modelPaths = new List<string>();
    // Names of the available TFJS models
    private List<string> modelNames = new List<string>();
    // Names of the available TFJS backends
    private List<string> tfjsBackends = new List<string> { "webgl" };


    // A class for reading in normalization stats from a JSON file
    class NormalizationStats { public float[] mean; public float[] std; }
    float[] mean = new float[] {0.485f, 0.456f, 0.406f };
    float[] std = new float[] { 0.229f, 0.224f, 0.225f };

    [System.Serializable]
    class ModelData { public string name; public string path; }
    [System.Serializable]
    class ModelList { public List<ModelData> models; }

    /// <summary>
    /// Stores the information for a single object
    /// </summary> 
    public struct Object
    {
        // The X coordinate for the top left bounding box corner
        public float x0;
        // The Y coordinate for the top left bounding box cornder
        public float y0;
        // The width of the bounding box
        public float width;
        // The height of the bounding box
        public float height;
        // The object class index for the detected object
        public int label;
        // The model confidence score for the object
        public float prob;

        public Object(float x0, float y0, float width, float height, int label, float prob)
        {
            this.x0 = x0;
            this.y0 = y0;
            this.width = width;
            this.height = height;
            this.label = label;
            this.prob = prob;
        }
    }

    public struct GridAndStride
    {
        public int grid0;
        public int grid1;
        public int stride;

        public GridAndStride(int grid0, int grid1, int stride)
        {
            this.grid0 = grid0;
            this.grid1 = grid1;
            this.stride = stride;
        }
    }

    // Stores information for the current list of detected objects
    //private List<Object> objectInfoArray = new List<Object>();
    private Object[] objectInfoArray;

    List<GridAndStride> grid_strides = new List<GridAndStride>();
    int[] strides = new int[] { 8, 16, 32 };
    float[] output_array;
    float scale_x;
    float scale_y;

    bool isInitialized;

    /// <summary>
    /// Initialize the selected webcam device
    /// </summary>
    /// <param name="deviceName">The name of the selected webcam device</param>
    private void InitializeWebcam(string deviceName)
    {
        // Stop any webcams already playing
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(deviceName, webcamDims.x, webcamDims.y, webcamFPS);

        // Start the webcam
        webcamTexture.Play();
        // Check if webcam is playing
        useWebcam = webcamTexture.isPlaying;
        // Update toggle value
        useWebcamToggle.SetIsOnWithoutNotify(useWebcam);

        Debug.Log(useWebcam ? "Webcam is playing" : "Webcam not playing, option disabled");
    }

    /// <summary>
    /// Resize and position an in-scene screen object
    /// </summary>
    private void InitializeScreen()
    {
        // Set the texture for the screen object
        screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture = useWebcam ? webcamTexture : imageTexture;
        // Set the screen dimensions
        screenDims = useWebcam ? new Vector2Int(webcamTexture.width, webcamTexture.height) : imageDims;

        // Flip the screen around the Y-Axis when using webcam
        float yRotation = useWebcam && mirrorScreen ? 180f : 0f;
        // Invert the scale value for the Z-Axis when using webcam
        float zScale = useWebcam && mirrorScreen ? -1f : 1f;

        // Set screen rotation
        screen.rotation = Quaternion.Euler(0, yRotation, 0);
        // Adjust the screen dimensions
        screen.localScale = new Vector3(screenDims.x, screenDims.y, zScale);

        // Adjust the screen position
        screen.position = new Vector3(screenDims.x / 2, screenDims.y / 2, 1);
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="json"></param>
    private void GetTFJSModels(string json)
    {
        ModelList modelList = JsonUtility.FromJson<ModelList>(json);
        foreach (ModelData model in modelList.models)
        {
            //Debug.Log($"{model.name}: {model.path}");
            modelNames.Add(model.name);
            string path = $"{Application.streamingAssetsPath}{model.path}";
            modelPaths.Add(path);
        }
        // Remove default dropdown options
        modelDropdown.ClearOptions();
        // Add TFJS model names to menu
        modelDropdown.AddOptions(modelNames);
        // Select the first option in the dropdown
        modelDropdown.SetValueWithoutNotify(0);
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="uri"></param>
    /// <returns></returns>
    IEnumerator GetRequest(string uri)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(uri))
        {
            // Request and wait for the desired page.
            yield return webRequest.SendWebRequest();

            string[] pages = uri.Split('/');
            int page = pages.Length - 1;

            switch (webRequest.result)
            {
                case UnityWebRequest.Result.ConnectionError:
                case UnityWebRequest.Result.DataProcessingError:
                    Debug.LogError(pages[page] + ": Error: " + webRequest.error);
                    break;
                case UnityWebRequest.Result.ProtocolError:
                    Debug.LogError(pages[page] + ": HTTP Error: " + webRequest.error);
                    break;
                case UnityWebRequest.Result.Success:
                    Debug.Log(pages[page] + ":\nReceived: " + webRequest.downloadHandler.text);

                    GetTFJSModels(webRequest.downloadHandler.text);
                    UpdateTFJSModel();
                    break;
            }
        }
    }


    /// <summary>
    /// Initialize the GUI dropdown list
    /// </summary>
    private void InitializeDropdown()
    {
        // Create list of webcam device names
        List<string> webcamNames = new List<string>();
        foreach (WebCamDevice device in webcamDevices) webcamNames.Add(device.name);

        // Remove default dropdown options
        webcamDropdown.ClearOptions();
        // Add webcam device names to dropdown menu
        webcamDropdown.AddOptions(webcamNames);
        // Set the value for the dropdown to the current webcam device
        webcamDropdown.SetValueWithoutNotify(webcamNames.IndexOf(currentWebcam));

        string modelListPath = $"{Application.streamingAssetsPath}/models.json";
        StartCoroutine(GetRequest(modelListPath));

        // Remove default dropdown options
        backendDropdown.ClearOptions();
        // Add TFJS backend names to menu
        backendDropdown.AddOptions(tfjsBackends);
        // Select the first option in the dropdown
        backendDropdown.SetValueWithoutNotify(0);
    }


    /// <summary>
    /// Resize and position the main camera based on an in-scene screen object
    /// </summary>
    /// <param name="screenDims">The dimensions of an in-scene screen object</param>
    private void InitializeCamera(Vector2Int screenDims, string cameraName = "Main Camera")
    {
        // Get a reference to the Main Camera GameObject
        GameObject camera = GameObject.Find(cameraName);
        // Adjust the camera position to account for updates to the screenDims
        camera.transform.position = new Vector3(screenDims.x / 2, screenDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        camera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the screenDims
        camera.GetComponent<Camera>().orthographicSize = screenDims.y / 2;
    }


    // Awake is called when the script instance is being loaded
    private void Awake()
    {
        WebGLPluginJS.GetExternalJS();
    }


    // Start is called before the first frame update
    void Start()
    {
        // Get the source image texture
        imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
        // Get the source image dimensions as a Vector2Int
        imageDims = new Vector2Int(imageTexture.width, imageTexture.height);

        // Initialize list of available webcam devices
        webcamDevices = WebCamTexture.devices;
        foreach (WebCamDevice device in webcamDevices) Debug.Log(device.name);
        currentWebcam = webcamDevices[0].name;
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        // Initialize webcam
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);

        // Initialize list of color maps from JSON file
        colormapList = JsonUtility.FromJson<ColorMapList>(colormapFile.text);
        // Initialize the list of colors
        colors = new Color[colormapList.items.Count];
        // Initialize the list of color textures
        colorTextures = new Texture2D[colormapList.items.Count];

        // Populate the color and color texture arrays
        for (int i = 0; i < colors.Length; i++)
        {
            // Create a new color object
            colors[i] = new Color(
                colormapList.items[i].color[0],
                colormapList.items[i].color[1],
                colormapList.items[i].color[2]);
            // Create a single-pixel texture
            colorTextures[i] = new Texture2D(1, 1);
            colorTextures[i].SetPixel(0, 0, colors[i]);
            colorTextures[i].Apply();

        }

        // Initialize the webcam dropdown list
        InitializeDropdown();


        WebGLPluginJS.SetTFJSBackend(tfjsBackends[backendDropdown.value]);
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    private Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        // Clamp the minimum dimension value to 64px
        targetDim = Mathf.Max(targetDim, 64);

        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions using the target minimum dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }


    /// <summary>
    /// Scale the latest bounding boxes to the display resolution
    /// </summary>
    public void ScaleBoundingBoxes()
    {
        // Process new detected objects
        for (int i = 0; i < objectInfoArray.Length; i++)
        {
            // The smallest dimension of the screen
            float minScreenDim = Mathf.Min(screen.transform.localScale.x, screen.transform.localScale.y);
            // The smallest input dimension
            int minInputDim = Mathf.Min(inputTextureCPU.width, inputTextureCPU.height);
            // Calculate the scale value between the in-game screen and input dimensions
            float minImgScale = minScreenDim / minInputDim;
            // Calculate the scale value between the in-game screen and display
            float displayScale = Screen.height / screen.transform.localScale.y;

            // Scale bounding box to in-game screen resolution and flip the bbox coordinates vertically
            float x0 = objectInfoArray[i].x0 * minImgScale;
            float y0 = (inputTextureCPU.height - objectInfoArray[i].y0) * minImgScale;
            float width = objectInfoArray[i].width * minImgScale;
            float height = objectInfoArray[i].height * minImgScale;

            // Mirror bounding box across screen
            if (mirrorScreen && useWebcam)
            {
                x0 = screen.transform.localScale.x - x0 - width;
            }

            // Scale bounding boxes to display resolution
            objectInfoArray[i].x0 = x0 * displayScale;
            objectInfoArray[i].y0 = y0 * displayScale;
            objectInfoArray[i].width = width * displayScale;
            objectInfoArray[i].height = height * displayScale;

            // Offset the bounding box coordinates based on the difference between the in-game screen and display
            objectInfoArray[i].x0 += (Screen.width - screen.transform.localScale.x * displayScale) / 2;
        }
    }



    List<GridAndStride> GenerateGridStrides(int height, int width, int[] strides)
    {
        List<GridAndStride> grid_strides = new List<GridAndStride>();

        // Iterate through each stride value
        foreach (int stride in strides)
        {
            // Calculate the grid dimensions
            int grid_height = height / stride;
            int grid_width = width / stride;

            Debug.Log($"Gride: {grid_height} x {grid_width}");

            // Store each combination of grid coordinates
            for (int g1 = 0; g1 < grid_height; g1++)
            {

                for (int g0 = 0; g0 < grid_width; g0++)
                {
                    grid_strides.Add(new GridAndStride(g0, g1, stride));
                }
            }
        }
        return grid_strides;
    }

    List<Object> GenerateYOLOXProposals(float[] model_output, int proposal_length, List<GridAndStride> grid_strides, float bbox_conf_thresh= 0.3f)
    {
        List<Object> proposals = new List<Object>();

        // Obtain the number of classes the model was trained to detect
        int num_classes = proposal_length - 5;

        for (int anchor_idx = 0; anchor_idx < grid_strides.Count; anchor_idx++)
        {
            // Get the current grid and stride values
            var grid0 = grid_strides[anchor_idx].grid0;
            var grid1 = grid_strides[anchor_idx].grid1;
            var stride = grid_strides[anchor_idx].stride;

            // Get the starting index for the current proposal
            var start_idx = anchor_idx * proposal_length;

            // Get the coordinates for the center of the predicted bounding box
            var x_center = (model_output[start_idx + 0] + grid0) * stride;
            var y_center = (model_output[start_idx + 1] + grid1) * stride;

            // Get the dimensions for the predicted bounding box
            var w = Mathf.Exp(model_output[start_idx + 2]) * stride;
            var h = Mathf.Exp(model_output[start_idx + 3]) * stride;

            // Calculate the coordinates for the upper left corner of the bounding box
            var x0 = x_center - w * 0.5f;
            var y0 = y_center - h * 0.5f;

            x0 /= scale_x;
            y0 /= scale_y;
            w /= scale_x;
            h /= scale_y;

            // Get the confidence score that an object is present
            var box_objectness = model_output[start_idx + 4];

            // Initialize object struct with bounding box information
            Object obj = new Object(x0, y0, w, h, 0, 0);

            // Find the object class with the highest confidence score
            for (int class_idx = 0; class_idx < num_classes; class_idx++)
            {
                // Get the confidence score for the current object class
                var box_cls_score = model_output[start_idx + 5 + class_idx];
                // Calculate the final confidence score for the object proposal
                var box_prob = box_objectness * box_cls_score;

                // Check for the highest confidence score
                if (box_prob > obj.prob)
                {
                    obj.label = class_idx;
                    obj.prob = box_prob;
                }
            }

            // Only add object proposals with high enough confidence scores
            if (obj.prob > bbox_conf_thresh) proposals.Add(obj);
        }

        // Sort the proposals based on the confidence score in descending order
        proposals = proposals.OrderByDescending(x => x.prob).ToList();

        return proposals;
    }

    float CalcUnionArea(Object a, Object b)
    {
        var x = Mathf.Min(a.x0, b.x0);
        var y = Mathf.Min(a.y0, b.y0);
        var w = Mathf.Max(a.x0 + a.width, b.x0+ b.width) - x;
        var h = Mathf.Max(a.y0 + a.height, b.y0 + b.height) - y;
        return w * h;
    }
    
    float CalcInterArea(Object a, Object b)
    {
        var x = Mathf.Max(a.x0, b.x0);
        var y = Mathf.Max(a.y0, b.y0);
        var w = Mathf.Min(a.x0 + a.width, b.x0 + b.width) - x;
        var h = Mathf.Min(a.y0 + a.height, b.y0 + b.height) - y;
        return w * h;
    }

    List<int> NMSSortedBoxes(List<Object> proposals, float nms_thresh= 0.45f)
    {
        List<int> proposal_indices = new List<int>();

        for (int i=0; i < proposals.Count; i++)
        {
            var a = proposals[i];
            bool keep = true;

            // Check if the current object proposal overlaps any selected objects too much
            foreach(int j in proposal_indices)
            {
                var b = proposals[j];

                // Calculate the area where the two object bounding boxes overlap
                var inter_area = CalcInterArea(a, b);

                // Calculate the union area of both bounding boxes
                var union_area = CalcUnionArea(a, b);

                // Ignore object proposals that overlap selected objects too much
                if (inter_area / union_area > nms_thresh) keep = false;
            }

            // Keep object proposals that do not overlap selected objects too much
            if (keep) proposal_indices.Add(i);
        }

        return proposal_indices;
    }

    // Update is called once per frame
    void Update()
    {
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        if (useWebcam)
        {
            // Initialize webcam if it is not already playing
            if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

            // Skip the rest of the method if the webcam is not initialized
            if (webcamTexture.width <= 16) return;

            // Make sure screen dimensions match webcam resolution when using webcam
            if (screenDims.x != webcamTexture.width)
            {
                // Resize and position the screen object using the source image dimensions
                InitializeScreen();
                // Resize and position the main camera using the source image dimensions
                InitializeCamera(screenDims);
            }
        }
        else if (webcamTexture && webcamTexture.isPlaying)
        {
            // Stop the current webcam
            webcamTexture.Stop();

            // Resize and position the screen object using the source image dimensions
            InitializeScreen();
            // Resize and position the main camera using the source image dimensions
            InitializeCamera(screenDims);
        }


        // Scale the source image resolution
        Vector2Int sourceDims = CalculateInputDims(screenDims, targetDim);
        Vector2Int inputDims = sourceDims;
        inputDims[0] = (inputDims[0] - inputDims[0] % 32) + 1;
        inputDims[1] = (inputDims[1] - inputDims[1] % 32) + 1;
        scale_x = inputDims[0] / (float)sourceDims[0];
        scale_y = inputDims[1] / (float)sourceDims[1];
        

        // Initialize the input texture with the calculated input dimensions
        inputTextureGPU = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGB32);

        if (!inputTextureCPU || inputTextureCPU.width != inputTextureGPU.width)
        {
            inputTextureCPU = new Texture2D(inputDims.x, inputDims.y, TextureFormat.RGB24, false);
            grid_strides = new List<GridAndStride>();
            grid_strides = GenerateGridStrides(inputDims[1], inputDims[0], strides);

            int output_size = grid_strides.Count * (colors.Length + 5);
            output_array = new float[output_size];
            WebGLPluginJS.UpdateOutputArray(output_array, output_size);
            Debug.Log($"Updating output array to {output_size}");
            Debug.Log($"Input Dims: {inputTextureCPU.width}x{inputTextureCPU.height}");
        }

        if (printDebugMessages) Debug.Log($"Input Dims: {inputTextureGPU.width}x{inputTextureGPU.height}");

        // Copy the source texture into model input texture
        Graphics.Blit((useWebcam ? webcamTexture : imageTexture), inputTextureGPU);

        // Download pixel data from GPU to CPU
        RenderTexture.active = inputTextureGPU;
        inputTextureCPU.ReadPixels(new Rect(0, 0, inputTextureGPU.width, inputTextureGPU.height), 0, 0);
        inputTextureCPU.Apply();

        int width = inputTextureCPU.width;
        int height = inputTextureCPU.height;
        int size = width * height * 3;
        isInitialized = WebGLPluginJS.PerformInference(inputTextureCPU.GetRawTextureData(), size, width, height);

        if (isInitialized == false)
        {
            // Release the input texture
            RenderTexture.ReleaseTemporary(inputTextureGPU);
            return;
        }

        List<Object> proposals = GenerateYOLOXProposals(output_array, colors.Length + 5, grid_strides, minConfidence);
        
        List<int> proposal_indices = NMSSortedBoxes(proposals);
        numObjects = proposal_indices.Count;
        objectInfoArray = new Object[numObjects];
        for(int i=0; i < objectInfoArray.Length; i++)
        {
            objectInfoArray[i] = proposals[proposal_indices[i]];
        }
        ScaleBoundingBoxes();
        
        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTextureGPU);
    }


    /// <summary>
    /// This method is called when the value for the webcam toggle changes
    /// </summary>
    /// <param name="useWebcam"></param>
    public void UpdateWebcamToggle(bool useWebcam)
    {
        this.useWebcam = useWebcam;
    }


    /// <summary>
    /// 
    /// </summary>
    public void UpdateTFJSBackend()
    {
        WebGLPluginJS.SetTFJSBackend(tfjsBackends[backendDropdown.value]);
    }


    /// <summary>
    /// The method is called when the selected value for the webcam dropdown changes
    /// </summary>
    public void UpdateWebcamDevice()
    {
        currentWebcam = webcamDevices[webcamDropdown.value].name;
        Debug.Log($"Selected Webcam: {currentWebcam}");
        // Initialize webcam if it is not already playing
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);
    }


    /// <summary>
    /// Update the minimum confidence score for keeping bounding box proposals
    /// </summary>
    /// <param name="slider"></param>
    public void UpdateConfidenceThreshold(Slider slider)
    {
        minConfidence = slider.value;
    }


    /// <summary>
    /// 
    /// </summary>
    public void UpdateTFJSModel()
    {
        WebGLPluginJS.InitTFJSModel(modelPaths[modelDropdown.value], mean, std);
    }


    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        // Initialize a rectangle for label text
        Rect labelRect = new Rect();
        // Initialize a rectangle for bounding boxes
        Rect boxRect = new Rect();

        GUIStyle labelStyle = new GUIStyle
        {
            fontSize = (int)(Screen.width * 11e-3)
        };
        labelStyle.alignment = TextAnchor.MiddleLeft;

        foreach (Object objectInfo in objectInfoArray)
        {
            if (!displayBoundingBoxes) break;

            // Skip object if label index is out of bounds
            if (objectInfo.label > colors.Length - 1) continue;

            // Get color for current class index
            Color color = colors[objectInfo.label];
            // Get label for current class index
            string name = colormapList.items[objectInfo.label].label;

            // Set bounding box coordinates
            boxRect.x = objectInfo.x0;
            boxRect.y = Screen.height - objectInfo.y0;
            // Set bounding box dimensions
            boxRect.width = objectInfo.width;
            boxRect.height = objectInfo.height;

            // Scale bounding box line width based on display resolution
            int lineWidth = (int)(Screen.width * 1.75e-3);
            // Render bounding box
            GUI.DrawTexture(
                position: boxRect,
                image: Texture2D.whiteTexture,
                scaleMode: ScaleMode.StretchToFill,
                alphaBlend: true,
                imageAspect: 0,
                color: color,
                borderWidth: lineWidth,
                borderRadius: 0);

            // Include class label and confidence score in label text
            string labelText = $" {name}: {(objectInfo.prob * 100).ToString("0.##")}%";

            // Initialize label GUI content
            GUIContent labelContent = new GUIContent(labelText);

            // Calculate the text size.
            Vector2 textSize = labelStyle.CalcSize(labelContent);

            // Set label text coordinates
            labelRect.x = objectInfo.x0;
            labelRect.y = Screen.height - objectInfo.y0 - textSize.y + lineWidth;

            // Set label text dimensions
            labelRect.width = Mathf.Max(textSize.x, objectInfo.width);
            labelRect.height = textSize.y;
            // Set label text and backgound color
            labelStyle.normal.textColor = color.grayscale > 0.5 ? Color.black : Color.white;
            labelStyle.normal.background = colorTextures[objectInfo.label];
            // Render label
            GUI.Label(labelRect, labelContent, labelStyle);

            Rect objectDot = new Rect();
            objectDot.height = lineWidth * 5;
            objectDot.width = lineWidth * 5;
            float radius = objectDot.width / 2;
            objectDot.x = (boxRect.x + boxRect.width / 2) - radius;
            objectDot.y = (boxRect.y + boxRect.height / 2) - radius;


            GUI.DrawTexture(
                position: objectDot,
                image: Texture2D.whiteTexture,
                scaleMode: ScaleMode.StretchToFill,
                alphaBlend: true,
                imageAspect: 0,
                color: color,
                borderWidth: radius,
                borderRadius: radius);

        }

        // Define styling information for GUI elements
        GUIStyle style = new GUIStyle
        {
            fontSize = (int)(Screen.width * (1f / (100f - fontScale)))
        };
        style.normal.textColor = textColor;

        // Define screen spaces for GUI elements
        Rect slot1 = new Rect(10, 10, 500, 500);
        Rect slot2 = new Rect(10, style.fontSize * 1.5f, 500, 500);

        string content = $"Objects Detected: {numObjects}";
        if (displayProposalCount) GUI.Label(slot1, new GUIContent(isInitialized ? content : "Loading Model..."), style);

        // Update framerate value
        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        // Adjust screen position when not showing predicted class
        Rect fpsRect = displayProposalCount ? slot2 : slot1;
        if (displayFPS) GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }
}
