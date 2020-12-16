const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegressionTensor {
  constructor(features, labels, options) {
    this.labels = tf.tensor(labels);
    this.features = this.processFeatures(features);
    this.mseHistory = [];
    this.bHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
      },
      options
    );
    this.weights = tf.zeros([this.features.shape[1], 1]); // [ [b], [M] ]
  }

  // features * ((features * weights) - labels)/n
  gradientDescent() {
    const currentGuess = this.features.matMul(this.weights);
    const differences = currentGuess.sub(this.labels);
    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));

    //this.weights.print();
    //console.log(`weights, M: ${this.weights.get(1, 0)}, b: ${this.weights.get(0, 0)}` );
  }

  // Batch Gradient Descent
  // Sochastic Gradient Descent
  batchGradientDescent(features, labels) {
    const currentGuess = features.matMul(this.weights);
    const differences = currentGuess.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  trainWithGradientDescent() {
    for (let i = 0; i < this.options.iterations; i++) {
      console.log(`=> Iter: ${i}`);
      this.bHistory.push(this.weights.get(0, 0));
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        console.log(`=> Iter: ${i}, batch: ${j}`);

        const { batchSize } = this.options;
        const startIndex = j * batchSize;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.batchGradientDescent(featureSlice, labelSlice);
      }

      this.bHistory.push(this.weights.get(0, 0));
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testLabels = tf.tensor(testLabels);
    testFeatures = this.processFeatures(testFeatures);
    const predictions = testFeatures.matMul(this.weights);
    console.log("predictions:");
    predictions.print();
    const ss_res = testLabels.sub(predictions).pow(2).sum().get();
    const ss_tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();
    const R2 = 1 - ss_res / ss_tot;
    console.log("R2:", R2);
    return R2;
  }

  processFeatures(features) {
    features = tf.tensor(features);
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standarize(features);
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standarize(features) {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    console.log("mse:", mse);
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }
    const lastValue = this.mseHistory[0];
    const secondValue = this.mseHistory[1];

    if (lastValue > secondValue) {
      this.options.learningRate /= 2.0;
    } else {
      this.options.learningRate *= 1.05;
    }
    console.log("learningRate:", this.options.learningRate);
  }
}

module.exports = LinearRegressionTensor;
