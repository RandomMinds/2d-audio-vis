var inputWav = process.argv[2], frameDir = process.argv[3] || 'frames';

var path = require('path');
var fse = require('fs-extra');
var ndarray = require('ndarray');
var ndarrayFft = require('ndarray-fft');
var ndarrayWav = require('ndarray-wav');
var savePixels = require('save-pixels');

function generate(width, height, valueFn) {
    var result = ndarray(new Float64Array(width*height), [width, height]);
    for (var x = 0; x < width; x++) {
        for (var y = 0; y < height; y++) {
            result.set(x, y, valueFn(x, y));
        }
    }
    return result;
}
function generateNoise(phase, ampFn, offset) {
    var width = phase.shape[0], height = phase.shape[1];
    var aspect = width/height;
    var amps = generate(width, height, function (x, y) {
        if (!x && !y) return offset || 0;
        if (x >= width/2) x -= width;
        if (y >= height/2) y -= width;
        var distance = Math.sqrt(x*x/aspect + y*y*aspect);
        return ampFn(distance)*width*height;
    });
    var real = generate(width, height, function (x, y) {
        return amps.get(x, y)*Math.cos(phase.get(x, y));
    });
    var imag = generate(width, height, function (x, y) {
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
}
function threshhold(noise, value, darknessValue) {
    value = value || 0;
    return generate(noise.shape[0], noise.shape[1], function (x, y) {
        var pixel = noise.get(x, y) - value;
        if (pixel < 0) return 255;
        if (!darknessValue) return 0;
        return 255 - Math.min(255, pixel*255/darknessValue);
    })
}

function normalise(data) {
    var sum = 0, sum2 = 0, total = 0;
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
            data.set(x, y, value*255);
            sum += value;
            sum2 += value*value;
            total++;
        }
    }
    var variance = sum2/total + sum*sum/total/total;
    console.log(variance, Math.sqrt(variance));
    return data;
}

function writePng(data, pngFile, callback) {
    var out = fse.createOutputStream(pngFile);
    savePixels(data, "png").pipe(out);
    out.on('end', function () {
        callback(null, pngFile);
    });
    out.on('error', function (error) {
        callback(error);
    });
}
var width = 1000, height = 500;

var phase = generate(width, height, function (x, y) {
    return Math.random()*2*Math.PI;
});
var noise = generateNoise(phase, function (distance) {
    var logDistance = Math.log(distance/2);
    return Math.exp(-logDistance*logDistance);
});

var pattern = threshhold(noise, 0);
writePng(pattern, 'noise.png', function (error) {
    if (error) throw error;
});

//var pattern = normalise(noise);
