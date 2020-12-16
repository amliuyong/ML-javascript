const tf = require("@tensorflow/tfjs");

class MutliLogisticRegression {
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
        descisionBoundary: 0.5,
      },
      options
    );
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  // Batch Gradient Descent
  // Sochastic Gradient Descent
  gradientDescent(features, labels) {
    const currentGuess = features.matMul(this.weights).softmax();
    const differences = currentGuess.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);
    return this.weights.sub(slopes.mul(this.options.learningRate));
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

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          return this.gradientDescent(featureSlice, labelSlice);
        }); // end tidy
      }

      this.bHistory.push(this.weights.get(0, 0));
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    console.log("predictions:");
    predictions.print();
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(testLabels).sum().get();
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
    const filler = variance.cast("bool").logicalNot().cast("float32");

    this.mean = mean;
    this.variance = variance.add(filler); // change 0 -> 1

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  // Corss Entropy
  recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log());
      const cost = termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .get(0, 0);

      console.log("cost:", cost);
      return cost;
    });
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

module.exports = MutliLogisticRegression;
