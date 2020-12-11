require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');
const LinearRegression = require('./LinearRegression');
const LinearRegressionTensor = require('./LinearRegressionTensor');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg']
});

console.log("features:", features);
console.log("labels:", labels);


// const linearRegressionV1 = new LinearRegression(features, labels, {
//     learningRate: 0.0001,
//     iterations: 10,
// });


const linearRegressionTensor = new LinearRegressionTensor(features, labels, {
    learningRate: 0.1,
    iterations: 100,
});

linearRegressionTensor.train();

console.log("features:")
linearRegressionTensor.features.print();

linearRegressionTensor.test(testFeatures, testLabels);


plot({
  x: linearRegressionTensor.mseHistory.reverse(),
  xLabel: 'Iteration #', 
  yLabel: 'MSE'
});


// plot({
//   x: linearRegressionTensor.bHistory,
//   y: linearRegressionTensor.mseHistory.reverse(),
//   xLabel: 'Value of B',
//   yLabel: 'MSE'
// });




