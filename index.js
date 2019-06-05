const tf = require('@tensorflow/tfjs-node-gpu')
const data = require('./data')
const model = require('./model')

const run = async ({ epochs, batchSize, modelSavePath }) => {
  // Load model.
  // ex:
  // await data.loadData()
  console.log('------------------------')
  console.log('Loading Data')
  await data.loadData()
  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  console.log('trainImages', trainImages)
  console.log('trainLabels', trainLabels)
  model.summary()
}

run({epochs: 50, batchSize: 10, modelSavePath: './'})

