// Define plugin functions
let plugin = {
    // Add additional JavaScript dependencies to the html page
    GetExternalJS: function () {

        // Add base TensorFlow.js dependencies
        let tfjs_script = document.createElement("script");
        tfjs_script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.20.0/dist/tf.min.js";
        document.head.appendChild(tfjs_script);

        // Add custom utility functions
        let script = document.createElement("script");
        script.src = "./StreamingAssets/utils.js";
        document.head.appendChild(script);
    },

    // Set the TFJS inference backend
    SetTFJSBackend: function (backend) {
        let backend_str = UTF8ToString(backend);
        try {
            tf.setBackend(backend_str).then(() => { });
            console.log(`Successfully set ${backend_str} backend.`);
        } catch (error) {
            console.log("Error occurred. Falling back to WebGL backend.");
            tf.setBackend('webgl');
        }
    },

    // Load a TFJS YOLOX model
    InitTFJSModel: async function (model_path, mean, std_dev) {

        // Convert bytes to the text
        let model_path_str = UTF8ToString(model_path);
        // Load the TensorFlow.js model at the provided file path
        this.model = await tf.loadGraphModel(model_path_str, { fromTFHub: false });

        // Check the model input shape
        const input_shape = this.model.inputs[0].shape;
        console.log(`Input Shape: ${input_shape}`);

        // Store normalization stats
        this.mean = new Float32Array(buffer, mean, 3);
        this.std_dev = new Float32Array(buffer, std_dev, 3);
    },

    // Update the array which stores the raw model output
    UpdateOutputArray: function (array_data, size) {
        delete this.output_array;
        this.output_array = new Float32Array(buffer, array_data, size);
        console.log(`New output size JS: ${this.output_array.length}`);
    },

    // Perform inference with the provided image data
    PerformInference: function (image_data, size, width, height) {

        // Only perform inference after loading a model
        if (typeof this.model == 'undefined') {
            console.log("Model not defined yet");
            return false;
        }

        // Initialize an array with the raw image data
        const uintArray = new Uint8ClampedArray(buffer, image_data, size, width, height);

        // Channels-last order
        const [input_array] = new Array(new Array());

        // Flip input image from Unity
        for (let row = height - 1; row >= 0; row--) {
            let slice = uintArray.slice(row * width * 3, (row * width * 3) + (width * 3));
            for (let col = 0; col < slice.length; col += 3) {
                input_array.push(((slice[col + 0] / 255.0) - this.mean[0]) / this.std_dev[0]);
                input_array.push(((slice[col + 1] / 255.0) - this.mean[1]) / this.std_dev[1]);
                input_array.push(((slice[col + 2] / 255.0) - this.mean[2]) / this.std_dev[2]);
            }
        }

        // Initialize the input array with the preprocessed input data
        const float32Data = Float32Array.from(input_array);
        const shape = [1, height, width, 3];

        // Pass preprocessed input to the model
        PerformInferenceAsync(this.model, float32Data, shape).then(output => {
            if (output_array.length == output.length) {
                this.output_array.set(output);
            }
            else {
                console.log(`Model output size JS: ${output.length}`);
                this.output_array.fill(0);
            }
        })
        return true;
    },
}

// Add plugin functions
mergeInto(LibraryManager.library, plugin);