const fs = require('fs')
const util = require('util')
const https = require('https')
const zlib = require('zlib')
const tf = require('@tensorflow/tfjs-node-gpu')

const readFile = util.promisify(fs.readFile)

const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte'
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte'
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte'
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte'
const IMAGE_HEADER_BYTES = 16
const IMAGE_HEIGHT = 28
const IMAGE_WIDTH = 28
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
const LABEL_HEADER_BYTES = 8
const LABEL_RECORD_BYTE = 1
const LABEL_FLAT_SIZE = 10

async function loadImages(filename) {
  const buffer = await downloadAndSave(filename)
  const headerBytes = IMAGE_HEADER_BYTES
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH
  const images = []
  let index = headerBytes
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes)
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++) / 255
    }
    images.push(array)
  }
  return images;
}

async function loadLabels(filename) {
  const buffer = await downloadAndSave(filename)
  const headerBytes = LABEL_HEADER_BYTES
  const recordBytes = LABEL_RECORD_BYTE
  const labels = []
  let index = headerBytes
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes)
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++)
    }
    labels.push(array)
  }
  return labels

}

async function downloadAndSave(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.gz`
    if (fs.existsSync(filename)) {
      return resolve(readFile(filename))
    }
    const file = fs.createWriteStream(filename)
    console.log(`  * Downloading from: ${url}`)
    https.get(url, response => {
      const unzip = zlib.createGunzip()
      response.pipe(unzip).pipe(file)
      unzip.on('end', () => {
        resolve(readFile(filename))
      })
    })
  })
}

class Dataset {
  constructor () {
    this.dataset
    this.trainSize = 0
    this.testSize = 0
  }

  async loadData () {
     this.dataset = await Promise.all([
      loadImages(TRAIN_IMAGES_FILE),
      loadLabels(TRAIN_LABELS_FILE),
      loadImages(TEST_IMAGES_FILE),
      loadLabels(TEST_LABELS_FILE)
    ])
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData) {
    let imagesIndex
    let labelsIndex
    if (isTrainingData) {
      imagesIndex = 0
      labelsIndex = 1
    } else {
      imagesIndex = 2
      labelsIndex = 3
    }
    const size = this.dataset[imagesIndex].length

    // Only create one big array to hold batch of images.
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape))
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]))

    let imageOffset = 0
    let labelOffset = 0
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset)
      labels.set(this.dataset[labelsIndex][i], labelOffset)
      imageOffset += IMAGE_FLAT_SIZE
      labelOffset += 1
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }

}

module.exports = new Dataset()