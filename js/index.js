import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import "core-js/stable";
import "regenerator-runtime/runtime";

async function run() {
    const MODEL_URL = '/karras2018iclr-celebahq-1024x1024.tfjs/model.json';

    const model = await loadGraphModel(MODEL_URL)
    const input = tf.randomNormal([1, 512]);
    let output = model.predict(input)

    output = tf.clipByValue(output, 0, 1.0);
    output = output.squeeze();

    var canvas = document.getElementById("myCanvas")
    await tf.browser.toPixels(output, canvas);
    console.log('done');
}

document.addEventListener('DOMContentLoaded', run);