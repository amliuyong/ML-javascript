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
