rem Convert a TensorFlow model to a TensorFlow.js model.
rem Convert to dist folder, as the parcel server will search for files there.
if not exist dist mkdir dist

tensorflowjs_converter ..\karras2018iclr-celebahq-1024x1024.tf dist/karras2018iclr-celebahq-1024x1024.tfjs