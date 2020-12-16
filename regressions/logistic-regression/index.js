require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const loadCSV = require("../load-csv");
const plot = require("node-remote-plot");
const LogisticRegression = require("./LogisticRegression");

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    converters: {
      passedemissions: (value) => {
        return value === 'TRUE' ? 1 : 0;
      },
    },
  }
);

//console.log(labels);

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10
});

regression.train();

regression.test(testFeatures, testLabels);

regression.predict([
   [130, 307, 1.75],
   [88, 97, 1.067]
]).print();

plot({
    x: regression.costHistory.reverse(),
    xLabel: 'Iteration #', 
    yLabel: 'Cross Entropy'
});





