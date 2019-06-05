console.log(tf)

function setup() {
  let canvas = createCanvas(280, 280);
  canvas.parent('sketch-holder')
  background(0, 0, 0)

}

function draw() {
  if (mouseIsPressed) {
    stroke(255)
    fill(255);
    ellipse(mouseX, mouseY, 24, 24);

  }
}


async function run() {
  const model = await tf.loadLayersModel('model.json')
  let canvas = document.getElementById('defaultCanvas0')

  let clearBtn = document.getElementById('clear')
  let predictBtn = document.getElementById('predict')
  let predictionDiv = document.getElementById('prediction')

  clearBtn.addEventListener('click', () => {
    background(0, 0, 0)
  })
  predictBtn.addEventListener('click', () => {
    let imgPixel = tf.browser.fromPixels(canvas, 1).reshape([-1, 280, 280, 1])
    let xx = tf.image.cropAndResize(
      imgPixel.toFloat(),
      [[0, 0, 1, 1]],
      [0],
      [28, 28]
    )
    let s = model.predict(xx)
    s.print()
    const logits = Array.from(s.dataSync())
    console.log('logits', logits)
    predictionDiv.innerText = 'Result: ' + s.argMax(-1).dataSync()[0]
    const buffer = new Uint8ClampedArray(28 * 28 * 4)
    const imageData = new ImageData(28, 28)
    const data = xx.dataSync()

    var pos = 0
    for (var i =0; i<28*28; i++) {
      buffer[pos] = data[i]
      buffer[pos +1] = data[i]
      buffer[pos +2] = data[i]
      buffer[pos +3] = 255
      pos+=4
    }

    var resultCanvas = document.getElementById('result-canvas')
    var ctx = resultCanvas.getContext('2d')
    imageData.data.set(buffer)
    ctx.putImageData(imageData, 0, 0);


  })
}


document.addEventListener("DOMContentLoaded", run, false);
