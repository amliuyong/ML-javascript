const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );

    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    // yHat =  m * X + b
    const currentGuessForMPG = this.features.map((row) => {
      return this.m * row[0] + this.b;
    });

    // Sum( yHat - Y ) * 2 / N
    const bSlop =
      (_.sum(
        currentGuessForMPG.map((guess, i) => {
          return guess - this.labels[i][0];
        })
      ) *
        2) /
      this.labels.length;

      // Sum( -1 * X * (Y - yHat)) * 2 / N
    const mSlop =
      (_.sum(
        currentGuessForMPG.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.labels.length;

      this.m = this.m - mSlop * this.options.learningRate;
      this.b = this.b - bSlop * this.options.learningRate;
      
      console.log(`\tmSlop: ${mSlop}, bSlop: ${bSlop}`)
    
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
      console.log(`=> Iter: ${i}, M: ${this.m}, B: ${this.b}`)
    }
  }
}

module.exports = LinearRegression;
