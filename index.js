const tf = require('@tensorflow/tfjs-node-gpu')
const data = require('./data')
const model = require('./model')

const run = async ({ epochs, batchSize }) => {
  await data.loadData()
  const {images: trainImages, labels: trainLabels} = data.getTrainData()
  model.summary()


  const validationSplit = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit
  })

  const {images: testImages, labels: testLabels} = data.getTestData()
  const evalOutput = model.evaluate(testImages, testLabels)

  let loss = evalOutput[0].dataSync()[0].toFixed(3)
  let accuracy = evalOutput[1].dataSync()[0].toFixed(3)
  console.log('----------------------------')
  console.log(`    Lost: ${loss}`)
  console.log(`Accuracy: ${accuracy}`)
  console.log('----------------------------')
  await model.save(`file://model.fit`);

}

run({epochs: 20, batchSize: 64})

