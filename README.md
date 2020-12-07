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

# KNN
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
