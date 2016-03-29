var inputWav = process.argv[2];
var frameDir = process.argv[3] || 'frames';
var frameRate = parseFloat(process.argv[4]) || 30;

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
        if (!x && !y) return 0;
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

function std(data) {
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
}

function normalise(data) {
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
        }
    }
    return data;
}

function writePng(data, pngFile, callback) {
    var out = fse.createOutputStream(pngFile);
    savePixels(data, "png").pipe(out);
    out.on('close', function () {
        callback(null, pngFile);
    });
    out.on('error', function (error) {
        callback(error);
    });
}
var width = 1280, height = 720;
/*
width /= 5;
height /= 5;
//*/

var phaseStart = generate(width, height, function (x, y) {
    return Math.random()*2*Math.PI;
});
var phaseRates = generate(width, height, function (x, y) {
    var t = Math.random() - Math.random();
    return t*5;
});

if (!inputWav) {
    var noise = generateNoise(phase, function (distance) {
        var logDistance = Math.log(distance/2);
        return Math.exp(-logDistance*logDistance);
    });
    var pattern = threshhold(noise, 0);
    writePng(pattern, 'noise.png', function (error) {
        if (error) throw error;
    });
    return;
}

var baseFreq = 40;
var threshholdAmp = 0, threshholdAmpDarkness = 0.001;
//var threshholdAmp = -1, threshholdAmpDarkness = 2;
var threshholdAmpFactor = 0, threshholdDarknessFactor = 0.01;
var timeOffset = 0.01;
function freqMultiplier(freq) {
    if (freq < 80) return freq/80;
    //if (freq > 4000) return 1 + freq/4000)/2;
    return 1;
}

ndarrayWav.open(inputWav, function (error, chunks) {
    if (error) throw error;
    var format = chunks.fmt;
    var sampleRate = format.sampleRate;
    var waveform = chunks.data;

    var durationSeconds = waveform.shape[1]/sampleRate;
    var frameTimes = [];
    for (var frameTime = 0; frameTime < durationSeconds; frameTime += 1/frameRate) {
        frameTimes.push(frameTime + timeOffset);
    }
    var frameIndexCounter = 0;
    var frameIndexTotal = frameTimes.length;
    var startEncodeMs = Date.now();
    function nextFrame() {
        if (!frameTimes.length) {
            console.log('\ndone');
            return;
        }
        var frameIndex = frameIndexCounter++;
        var frameFile = '' + frameIndex;
        while (frameFile.length < 5) {
            frameFile = '0' + frameFile;
        }
        frameFile = path.join(frameDir, 'frame' + frameFile + '.png');

        var frameTime = frameTimes.shift();
        var framePreTime = Math.max(1/frameRate, 0.1);
        var framePostTime = Math.max(1/frameRate, 0.03);
        var extractMidSample = frameTime*sampleRate;
        var extractStartSample = Math.round((frameTime - framePreTime)*sampleRate);
        var extractEndSample = Math.round((frameTime + framePostTime)*sampleRate);

        var fftSize = 256;
        while (fftSize < (extractEndSample - extractStartSample) || fftSize < width || fftSize < height) {
            fftSize *= 2;
        }
        var extractAmp2 = 0;
        var extract = generate(2, fftSize, function (channel, index) {
            var sampleIndex = index + extractStartSample;
            if (sampleIndex < 0) {
                return 0;
            }
            if (sampleIndex >= extractEndSample || sampleIndex >= waveform.shape[1]) {
                return 0;
            }
            var value = waveform.get(channel, sampleIndex);
            var offsetSample = sampleIndex - extractMidSample;
            var windowValue = 1;
            if (offsetSample < 0) {
                // Pre-window
                var windowRatio = -offsetSample/(extractMidSample - extractStartSample);
                windowValue = 1 + Math.cos(windowRatio*Math.PI);
            } else {
                // Post-window
                var windowRatio = offsetSample/(extractEndSample - extractMidSample);
                windowValue = 1 + Math.cos(windowRatio*Math.PI);
            }
            extractAmp2 += value*value*windowValue;
            return value*windowValue;
        });
        var extractAmp = Math.sqrt(extractAmp2/(extractEndSample - extractStartSample));
        ndarrayFft(1, extract.pick(0), extract.pick(1));

        function ampFunction(distance) {
            var freq = distance*baseFreq;
            var index = Math.round(freq/sampleRate*fftSize);
            if (index >= fftSize/2) return 0;
            var index2 = fftSize - index;
            var leftReal = (extract.pick(0, index) + extract.pick(0, index2))/2;
            var leftImag = (extract.pick(1, index) - extract.pick(1, index2))/2;
            var rightReal = (extract.pick(1, index) + extract.pick(1, index2))/2;
            var rightImag = (-extract.pick(0, index) + extract.pick(0, index2))/2;

            var leftMag2 = leftReal*leftReal + leftImag*leftImag;
            var rightMag2 = rightReal*rightReal + rightImag*rightImag;
            var mag = Math.sqrt((leftMag2 + rightMag2)/2);
            return mag/fftSize*freqMultiplier(freq);
        }

        var phase = generate(width, height, function (x, y) {
            return phaseStart.get(x, y) + frameTime*phaseRates.get(x, y);
        });
        var noise = generateNoise(phase, function (distance) {
            return ampFunction(distance);
        });
        var pattern = threshhold(noise, threshholdAmp + extractAmp*threshholdAmpFactor, threshholdAmpDarkness + extractAmp*threshholdDarknessFactor);
        writePng(pattern, frameFile, function (error) {
            if (error) throw error;
            var ratio = frameIndexCounter/frameIndexTotal;
            var endEstimateMs = (Date.now() - startEncodeMs)*(1 - ratio)/ratio;
            var endEstimateH = Math.floor(endEstimateMs/1000/60/60);
            var endEstimateM = Math.floor(endEstimateMs/1000/60)%60;
            var endEstimateS = Math.floor(endEstimateMs/1000)%60;
            process.stdout.clearLine();
            process.stdout.write('\rframe ' + frameIndexCounter + '/' + frameIndexTotal + '\tremaining: ' + endEstimateH + 'h' + endEstimateM + 'm' + endEstimateS + 's');
            nextFrame();
        });
    }
    //frameTimes[0] = 15;
    nextFrame();
});
