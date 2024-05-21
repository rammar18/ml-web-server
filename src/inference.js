// package tensorflow js
const tfjs = require('@tensorflow/tfjs-node');

// fungsi load model tensorflow js
function loadModel(){
    const modelUrl = "file://models/model.json";
    return tfjs.loadLayersModel(modelUrl);
}

// fungsi prediksi data (berupa imagebuffer)
function predict(model, imageBuffer){
    const tensor = tfjs.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([150, 150])
        .expandDims()
        .toFloat();

    return model.predict(tensor).data();
}

module.exports = { loadModel, predict };