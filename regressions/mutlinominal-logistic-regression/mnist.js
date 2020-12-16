require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
const loadCSV = require("../load-csv");
const plot = require("node-remote-plot");
const mnist = require("mnist-data");

const MutliLogisticRegression = require("./MutilLogisticRegression");

let mnistData = mnist.training(0, 60000);
const features = mnistData.images.values.map((image) => _.flatMap(image));
const encodedLabels = mnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});
mnistData = null;

//console.log(encodedLabels);

const regression = new MutliLogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
});

regression.train();

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("accuracy:", accuracy);


plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iteration #', 
  yLabel: 'Cross Entropy'
});

// node --max-old-space-size=4096 mnist.js

