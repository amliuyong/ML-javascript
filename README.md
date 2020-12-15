# ML Kits

Starter projects for learning about Machine Learning.

## Downloading

There are two ways to download this repository - either as a zip or by using git.

### Zip Download

To download this project as a zip file, find the green 'Clone or Download' button on the top right hand side. Click that button, then download this project as a zip file.

Once downloaded extract the zip file to your local computer.

### Git Download

To download this project using git, run the following command at your terminal:

```
git clone https://github.com/StephenGrider/MLKits.git
```

# lodash & JSPlaygrounds

- https://lodash.com/
- https://stephengrider.github.io/JSPlaygrounds/

```javascript

const numbers = [
    [10, 5],
    [11, 2],
    [9, 16],
    [10, -1]
  ];

const sorted = _.sortBy(numbers, (row) => row[1]);

const mapped = _.map(sorted, row => row[1]);

_.chain(numbers)
.sortBy( (row) => row[1])
.map(row => row[1])
.value();


```

# KNN by lodash
```javascript

const outputs = [];
//import _ from "lodash";

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  //console.log(outputs);
}

function distance(pointA, pointB) {
  // pointA = [300, 0.5, 16], pointB = [120, 0.45, 16],
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}

function runAnalysis_0() {
  // Write code here to analyze stuff
  const testSetSize = 100;
  const [testSet, trainSet] = splitDataset(minMax(outputs, 3), testSetSize);

  // let numberCorrect = 0;
  // for(let i =0; i< testSet.length; i++) {
  //   const bucket =  knn(trainSet, testSet[i][0]);
  //   console.log(bucket, testSet[i][3]);

  //   if (bucket ===  testSet[i][3]) {
  //     numberCorrect++;
  //   }
  // }

  // console.log('Accuracy:', numberCorrect /testSetSize );

  _.range(1, 20).forEach((k) => {
    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) =>
          knn(trainSet, _.initial(testPoint), k) === _.last(testPoint)
      )
      .size()
      .divide(testSetSize)
      .value();

    console.log(`k=${k}, Accuracy:${accuracy}`);
  });
}




function runAnalysis() {
  // Write code here to analyze stuff
  const testSetSize = 100;
  const k = 10;

  _.range(0, 3).forEach(feature => {

    const data = _.map(outputs, row => [row[feature], _.last(row)]);

    const [testSet, trainSet] = splitDataset(minMax(data, 1), testSetSize);

    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) =>
          knn(trainSet, _.initial(testPoint), k) === _.last(testPoint)
      )
      .size()
      .divide(testSetSize)
      .value();

    console.log(`feature=${feature}, Accuracy:${accuracy}`);
  });
}


//point = [300, 0.5, 16]
//row = [300, 0.5, 16, 5]
function knn(dataset, point, k) {
  return _.chain(dataset)
    .map((row) => {
      return [distance(_.initial(row), point), _.last(row)];
    })
    .sortBy((row) => row[0])
    .slice(0, k)
    .countBy((row) => row[1])
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function splitDataset(data, testCount) {
  const shuffed = _.shuffle(data);
  const testSet = _.slice(shuffed, 0, testCount);
  const trainSet = _.slice(shuffed, testCount);
  return [testSet, trainSet];
}

// Normalize data
function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const cloumn = clonedData.map((row) => row[i]);
    const min = _.min(cloumn);
    const max = _.max(cloumn);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }

  return clonedData;
}


```
# TesnorFlow

## tensor
```javascript

const data = tf.tensor(
  [
  [1,2,3],
  [4,5,6]
  ]
);
const otherData = tf.tensor(
[
  [4,5,6],
  [6,7,8] 
]
);

data.add(otherData)

data.div(otherData)

data.sub(otherData)

// for debug
data.print()


// get data from tensor
const data = tf.tensor([10, 20, 30]);
data.get(0);

const data2D = tf.tensor([[10, 20, 30],
                          [3, 5, 9]]);

data2D.get(1, 1);

data2D.set(1, 1, 60); // NO SE T!!!!


```
## slice
```javascript

const data = tf.tensor([
   [10, 20, 30 ],
   [10, 21, 30 ],
   [10, 22, 30 ],
   [10, 23, 30 ],
   [10, 24, 30 ],
   [10, 25, 30 ],
   [10, 26, 30 ],
   [10, 27, 30 ]  
  ]);

// slice(startIndex, size) 

data.slice([0, 1], [8, 1]); 
// => [[20], [21], [22], [23], [24], [25], [26], [27]]

data.shape // [8,3]

data.slice([0, 1], [data.shape[0], 1]); 
// => [[20], [21], [22], [23], [24], [25], [26], [27]]

data.slice([1, 1], [-1, 1]); 
// => [[21], [22], [23], [24], [25], [26], [27]]

```

## concat

```javascript

const tensorA = tf.tensor([
  [1, 2, 3],
  [4, 5, 6]
]);

const tensorB = tf.tensor([
  [12, 22, 32],
  [42, 52, 62]
]);


tensorA.concat(tensorB)
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [12, 22, 32], [42, 52, 62]]

tensorA.concat(tensorB, 0)
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [12, 22, 32], [42, 52, 62]]


tensorA.concat(tensorB, 1)
// [[1, 2, 3, 12, 22, 32], [4, 5, 6, 42, 52, 62]]

```

## sum, expandDims, concat
```javascript

const jumpData = tf.tensor([
   [70, 70, 70],
   [70, 70, 70],
   [70, 70, 70],
   [70, 70, 70],

]);

const playerData = tf.tensor([
  [1, 160],
  [2, 160],
  [3, 160],
  [4, 160],
]);


// what I want: 
[
 [210, 1, 160], 
 [210, 2, 160], 
 [210, 3, 160], 
 [210, 4, 160]
 ]


// Method 1
jumpData.sum(1);
//=> [210, 210, 210, 210]
jumpData.sum(1, true)
// => [[210], [210], [210], [210]]
playerData.shape // [4,2]
jumpData.sum(1, true).concat(playerData, 1);
// => [[210, 1, 160], [210, 2, 160], [210, 3, 160], [210, 4, 160]]

// Mothed 2
jumpData.sum(1); // [210, 210, 210, 210]
jumpData.sum(1).expandDims() //[[210, 210, 210, 210],]
jumpData.sum(1).expandDims(1) // [[210], [210], [210], [210]]
jumpData.sum(1).expandDims(1).concat(playerData, 1);

```

## Broadcasting

Can do Broadcasting

- shape[3] + shape[1]  

- shape[2, 3] + shape[2, 1]

- shape[2,3,2] + shape[3, 1]

Canot do Broadcasting

 - shape[2,3,2] + shape[2, 1]

## standardization with tensorflow

```javascript

const numbers = tf.tensor([
   [1, 2],
   [2, 3],
   [4, 5],
   [6, 7]  
]);

//const {mean, variance} = tf.moments(numbers);

//mean // 3.75
//variance //3.9375


const {mean, variance} = tf.moments(numbers, 0);
mean  //[3.25, 4.25]
variance // [3.6875, 3.6875]

numbers.sub(mean).div(variance.pow(0.5))
//
//[[-1.1717002, -1.1717002], 
// [-0.6509446, -0.6509446], 
// [0.3905667 , 0.3905667 ], 
// [1.432078 , 1.432078 ]]
//

```


## KNN with tesnsorflow

```javascript

const features = tf.tensor([
   [-121.1, 47.1 ],
   [-121.2, 47.2 ],
   [-121.3, 47.3 ],
   [-121.4, 47.5 ],
   [-121.5, 47.6 ],
   [-121.6, 47.7 ],
  ]);

const labels = tf.tensor([
   [201],
   [202],
   [203],
   [204],
   [205],
   [206]
  ]);

const predictionPoint = tf.tensor([-120, 48]);

const distances = features.sub(predictionPoint)
  .pow(2)
  .sum(1) // [1.4212668, 1.4422175, 1.4764854, 1.4866083, 1.5524179, 1.6278803]
  .pow(0.5);

distances.expandDims(1).concat(labels, 1);
//[[1.4212668, 201], 
// [1.4422175, 202], 
// [1.4764854, 203], 
// [1.4866083, 204], 
// [1.5524179, 205], 
// [1.6278803, 206]]

distances.expandDims(1).concat(labels, 1)
  .unstack()[0] // [1.4212668, 201]


const unsortedTesnsorArray = distances.expandDims(1).concat(labels, 1)
  .unstack() // Array of Tensor

const k = 3;

const topK = unsortedTesnsorArray.sort((a,b) => {
  return a.get(0) - b.get(0);
})
.slice(0, k);

const predictValue = topK.reduce((acc, pair) => {
    return acc + pair.get(1)
}, 0) / k;

predictValue;

```
## KNN with tesnsorflow - real word example

```javascript

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)               // standardization
      .div(variance.pow(0.5))  // standardization
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => {
        return a.get(0) - b.get(0);
      })
      .slice(0, k)
      .reduce((acc, pair) => {
        return acc + pair.get(1);
      }, 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  }
);

console.log("testFeatures:", testFeatures);
console.log("testLabels:", testLabels);

features = tf.tensor(features);
labels = tf.tensor(labels);

// testFeatures = tf.tensor(testFeatures);
// testLabels = tf.tensor(testLabels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

const err = Math.abs(testLabels[0][0] - result) / testLabels[0][0];

console.log("Guess", result, testLabels[0][0]);
console.log("err:", err * 100);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = Math.abs(testLabels[i][0] - result) / testLabels[i][0];
  console.log("err:", err * 100);
});

```
# LinearRegression

## LinearRegression using Tensor - gradientDescent/standarize/updateLearningRate
```javascript
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

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      console.log(`=> Iter: ${i}`);
      this.bHistory.push(this.weights.get(0, 0));
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate();
    }
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

```
## train LinearRegression and plot
```javascript

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

```
##  Batch Gradient Descent and Sochastic Gradient Descent(batchsize = 1)

```javascript
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


```
## predict

```javascript

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
    iterations: 3,
    batchSize: 10
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

//['horsepower', 'weight', 'displacement']
console.log("predict Value: ");
linearRegressionTensor.predict([
  [ 120, 2, 380],
  [ 125, 2.1, 420]
]).print();

```

# LogisticRegression

```javascript

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

```

```javascript
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
```

# MutliLogisticRegression

```javascript
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
        descisionBoundary: 0.5
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
    .softmax()
    .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    console.log("predictions:");
    predictions.print();
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(testLabels).sum().get();
    const correctPercentage = (predictions.shape[0] - incorrect) / predictions.shape[0];

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

module.exports = MutliLogisticRegression;


```
```javasscript
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

```
