const tf = require("@tensorflow/tfjs");

class LogisticRegression {
  constructor(features, labels, options) {
    this.labels = tf.tensor(labels);
    this.features = this.processFeatures(features);
    this.costHistory = [];
    this.bHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
        descisionBoundary: 0.5
      },
      options
    );
    this.weights = tf.zeros([this.features.shape[1], 1]); // [ [b], [M] ]
  }

  // Batch Gradient Descent
  // Sochastic Gradient Descent
  gradientDescent(features, labels) {
    const currentGuess = features.matMul(this.weights).sigmoid();
    const differences = currentGuess.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        //console.log(`=> Iter: ${i}, batch: ${j}`);
        const { batchSize } = this.options;
        const startIndex = j * batchSize;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }

      this.bHistory.push(this.weights.get(0, 0));
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
    .matMul(this.weights)
    .sigmoid()
    .greater(this.options.descisionBoundary)
    .cast('float32');
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    console.log("predictions:");
    predictions.print();
    testLabels = tf.tensor(testLabels);
    const incorrect = predictions.sub(testLabels).abs().sum().get();
    const correctPercentage =
      (predictions.shape[0] - incorrect) / predictions.shape[0];
    console.log("correctPercentage", correctPercentage);
    return correctPercentage;
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

  // Corss Entropy
  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels
    .transpose()
    .matMul(guesses.log());

    const termTwo = this.labels
    .mul(-1)
    .add(1)
    .transpose()
    .matMul(
        guesses.mul(-1).add(1).log()
    );
    const cost = termOne.add(termTwo)
    .div(this.features.shape[0])
    .mul(-1)
    .get(0, 0);

    console.log("cost:", cost);
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }
    const lastValue = this.costHistory[0];
    const secondValue = this.costHistory[1];

    if (lastValue > secondValue) {
      this.options.learningRate /= 2.0;
    } else {
      this.options.learningRate *= 1.05;
    }
    console.log("learningRate:", this.options.learningRate);
  }
}

module.exports = LogisticRegression;
