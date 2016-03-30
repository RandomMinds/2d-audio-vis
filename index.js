var inputWav = process.argv[2];
var frameDir = process.argv[3] || 'frames';
var frameRate = parseFloat(process.argv[4]) || 30;
var width = parseFloat(process.argv[5]) || 1280;
var height = parseFloat(process.argv[6]) || 720;

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
function threshholdProportion(noise, targetProportion) {
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
    return generate(noise.shape[0], noise.shape[1], function (x, y) {
        var pixel = noise.get(x, y);
        if (pixel >= value) return 255;
        return 0;
    });
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
var timeOffset = 0.015;
function freqMultiplier(freq) {
    if (freq < 80) return freq/80;
    //if (freq > 4000) return 1 + freq/4000)/2;
    return 1;
}
var ampMultiplier = 0.4, ampPower = 0.5;
function proportionCurve(v) {
    return 1 - Math.exp(-Math.pow(v, ampPower)*ampMultiplier);
}

ndarrayWav.open(inputWav, function (error, chunks) {
    if (error) throw error;
    var format = chunks.fmt;
    var sampleRate = format.sampleRate;
    var waveform = chunks.data;

    var waveformStd = std(waveform);

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
        function getSpectrum(width) {
            var extractMidSample = frameTime*sampleRate - width*0.1;
            var framePreTime = Math.max(1/frameRate, width*0.7);
            var framePostTime = Math.max(1/frameRate, width*0.3);
            var extractStartSample = Math.round(extractMidSample - framePreTime*sampleRate);
            var extractEndSample = Math.round(extractMidSample + framePostTime*sampleRate);

            var fftSize = 256;
            while (fftSize < (extractEndSample - extractStartSample) || fftSize < width || fftSize < height) {
                fftSize *= 2;
            }
            fftSize *= 2;   // Padding
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
                return value*windowValue;
            });
            ndarrayFft(1, extract.pick(0), extract.pick(1));

            return function ampFunction(distance) {
                var freq = distance*baseFreq;
                var index = Math.round(freq/sampleRate*fftSize);
                if (index >= fftSize/2 || index < 1) return 0;
                var index2 = fftSize - index;
                var leftReal = (extract.pick(0, index) + extract.pick(0, index2))/2;
                var leftImag = (extract.pick(1, index) - extract.pick(1, index2))/2;
                var rightReal = (extract.pick(1, index) + extract.pick(1, index2))/2;
                var rightImag = (-extract.pick(0, index) + extract.pick(0, index2))/2;

                var leftMag2 = leftReal*leftReal + leftImag*leftImag;
                var rightMag2 = rightReal*rightReal + rightImag*rightImag;
                var mag = Math.sqrt((leftMag2 + rightMag2)/2);
                return mag/fftSize*freqMultiplier(freq);
            };
        }

        /*
        // Use different widths of window for different frequencies
        var ampFunctions = {}, minCycles = 10;
        var minDuration = 0.015, maxDuration = 0.05;
        var ampFunction = function (distance) {
            var freq = Math.max(baseFreq, distance*baseFreq);
            var freqWidth = minCycles/freq;
            var spectrumWidth = Math.max(2/frameRate, minDuration);
            var prevSpectrumWidth = spectrumWidth;
            while (spectrumWidth < freqWidth && spectrumWidth < maxDuration) {
                prevSpectrumWidth = spectrumWidth;
                spectrumWidth *= 2;
            }
            spectrumWidth = Math.min(maxDuration, spectrumWidth);
            prevSpectrumWidth = Math.min(maxDuration, prevSpectrumWidth);
            if (!ampFunctions[spectrumWidth]) {
                ampFunctions[spectrumWidth] = getSpectrum(spectrumWidth);
            }
            if (!ampFunctions[prevSpectrumWidth]) {
                ampFunctions[prevSpectrumWidth] = getSpectrum(prevSpectrumWidth);
            }
            if (spectrumWidth === prevSpectrumWidth) {
                return ampFunctions[spectrumWidth](distance);
            }
            // Linearly interpolate between spectra of different lengths
            var ratio = (freqWidth - prevSpectrumWidth)/(spectrumWidth - prevSpectrumWidth);
            ratio = Math.max(0, Math.min(1, ratio));
            return ampFunctions[prevSpectrumWidth](distance)*(1 - ratio)
                + ampFunctions[spectrumWidth](distance)*ratio;
        }
        /*/

        // Override - single duration for all elements
        var ampFunction = getSpectrum(0.1);
        //*/

        var extractAmp2 = 0, extractAmpStep = 0.1;
        for (var dist = 0; dist < Math.sqrt(width*height); dist += extractAmpStep) {
            extractAmp2 += ampFunction(dist)*extractAmpStep;
        }
        var extractAmp = Math.sqrt(extractAmp2);

        var phase = generate(width, height, function (x, y) {
            return phaseStart.get(x, y) + frameTime*phaseRates.get(x, y);
        });
        var noise = generateNoise(phase, function (distance) {
            return ampFunction(distance);
        });
        var pattern = threshholdProportion(noise, proportionCurve(extractAmp/waveformStd));
        //var pattern = threshhold(noise, threshholdAmp + extractAmp*threshholdAmpFactor, threshholdAmpDarkness + extractAmp*threshholdDarknessFactor);
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
