var path = require('path');
var fse = require('fs-extra');
var ndarray = require('ndarray');
var ndarrayFft = require('ndarray-fft');
var ndarrayWav = require('ndarray-wav');
var getPixels = require('get-pixels');
var savePixels = require('save-pixels');

var api = module.exports = {
    generate: function generate(width, height, valueFn) {
        var result = ndarray(new Float64Array(width*height), [width, height]);
        for (var x = 0; x < width; x++) {
            for (var y = 0; y < height; y++) {
                result.set(x, y, valueFn(x, y));
            }
        }
        return result;
    },
    generateNoise: function generateNoise(phase, ampFn, offset) {
        var width = phase.shape[0], height = phase.shape[1];
        var aspect = width/height;
        var amps = api.generate(width, height, function (x, y) {
            if (!x && !y) return 0;
            if (x >= width/2) x -= width;
            if (y >= height/2) y -= height;
            var distance = Math.sqrt(x*x/aspect + y*y*aspect);
            return ampFn(distance)*width*height;
        });
        var real = api.generate(width, height, function (x, y) {
            return amps.get(x, y)*Math.cos(phase.get(x, y));
        });
        var imag = api.generate(width, height, function (x, y) {
            return amps.get(x, y)*Math.sin(phase.get(x, y));
        });
        // 2D IFFT
        for (var x = 0; x < width; x++) {
            ndarrayFft(-1, real.pick(x), imag.pick(x));
        }
        for (var y = 0; y < height; y++) {
            ndarrayFft(-1, real.pick(null, y), imag.pick(null, y));
        }
        return real;
    },
    threshholdValue: function threshhold(noise, value, darknessValue) {
        value = value || 0;
        return api.generate(noise.shape[0], noise.shape[1], function (x, y) {
            var pixel = noise.get(x, y) - value;
            if (pixel < 0) return 255;
            if (!darknessValue) return 0;
            return 255 - Math.min(255, pixel*255/darknessValue);
        })
    },
    threshholdProportion: function threshholdProportion(noise, targetProportion) {
        var values = [];
        for (var x = 0; x < noise.shape[0]; x++) {
            for (var y = 0; y < noise.shape[1]; y++) {
                values.push(noise.get(x, y));
            }
        }
        values.sort(function (a, b) {
            return a - b;
        });

        var valueIndex = Math.floor(targetProportion*values.length);
        var value = values[valueIndex];
        return api.generate(noise.shape[0], noise.shape[1], function (x, y) {
            var pixel = noise.get(x, y);
            if (pixel >= value) return 255;
            return 0;
        });
    },
    std: function std(data) {
        var sum = 0, sum2 = 0, total = 0;
        for (var x = 0; x < data.shape[0]; x++) {
            for (var y = 0; y < data.shape[1]; y++) {
                var value = data.get(x, y);
                sum += value;
                sum2 += value*value;
                total++;
            }
        }
        var variance = sum2/total + sum*sum/total/total;
        return Math.sqrt(variance);
    },
    mean: function mean(data) {
        var sum = 0, total = 0;
        for (var x = 0; x < data.shape[0]; x++) {
            for (var y = 0; y < data.shape[1]; y++) {
                var value = data.get(x, y);
                sum += value;
                total++;
            }
        }
        return sum/total;
    },
    normalise: function normalise(data, low, high) {
        if (typeof low !== 'number') low = 0;
        if (typeof high !== 'number') high = 255;

        var max = data.get(0, 0), min = max - 1e-20;;
        for (var x = 0; x < data.shape[0]; x++) {
            for (var y = 0; y < data.shape[1]; y++) {
                max = Math.max(max, data.get(x, y));
                min = Math.min(min, data.get(x, y));
            }
        }
        for (var x = 0; x < data.shape[0]; x++) {
            for (var y = 0; y < data.shape[1]; y++) {
                var value = data.get(x, y);
                value = (value - min)/(max - min);
                data.set(x, y, low + value*(high - low));
            }
        }
        return data;
    },
    writePng: function writePng(data, pngFile, callback) {
        var out = fse.createOutputStream(pngFile);
        savePixels(data, "png").pipe(out);
        out.on('close', function () {
            callback(null, pngFile);
        });
        out.on('error', function (error) {
            callback(error);
        });
    }
};
