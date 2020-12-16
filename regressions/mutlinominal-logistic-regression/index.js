require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
const loadCSV = require("../load-csv");
const plot = require("node-remote-plot");
const MutliLogisticRegression = require("./MutilLogisticRegression");

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["mpg"],
    converters: {
      mpg: (value) => {
        const mpg = parseFloat(value);
        if (mpg < 15) {
          return [1, 0, 0];
        } else if (mpg < 30) {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      },
    },
  }
);

//console.log(_.flatMap(labels));

labels = _.flatMap(labels);
testLabels = _.flatMap(testLabels);

const regression = new MutliLogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

console.log("weights:");
regression.weights.print();

regression.train();

console.log("after train weights:");
regression.weights.print();

regression.predict([
  [215, 440, 2.16],
  [150, 200, 2.223]
]).print();

regression.test(testFeatures, testLabels);


