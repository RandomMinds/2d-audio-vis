var inputWav = process.argv[2];
var frameDir = process.argv[3] || 'frames';
var frameRate = parseFloat(process.argv[4]) || 30;
var widthOrImage = process.argv[5];
var heightArg = process.argv[6];
var width, height;

var path = require('path');
var fse = require('fs-extra');
var ndarray = require('ndarray');
var ndarrayFft = require('ndarray-fft');
var ndarrayWav = require('ndarray-wav');
var getPixels = require('get-pixels');
var savePixels = require('save-pixels');

var common = require('./common');

function getBackgroundImage(callback) {
    if (widthOrImage && !heightArg) {
        getPixels(widthOrImage, function (error, pixels) {
            if (error) return callback(error);
            width = pixels.shape[0];
            height = pixels.shape[1];
            var greyscale = common.generate(width, height, function (x, y) {
                var sum = 0, total = 0;
                if (pixels.shape[3]) {
                    for (var f = 0; f < pixels.shape[3]; f++) {
                        for (var c = 0; c < pixels.shape[2]; c++) {
                            sum += pixels.get(x, y, c, f);
                            total++;
                        }
                    }
                    return sum/total;
                } else if (pixels.shape[2]) {
                    for (var c = 0; c < pixels.shape[2]; c++) {
                        sum += pixels.get(x, y, c);
                        total++;
                    }
                    return sum/total;
                } else {
                    return pixels.get(x, y);
                }
            });
            var m = common.mean(greyscale), s = Math.max(common.std(greyscale), 1e-20);
            for (var x = 0; x < width; x++) {
                for (var y = 0; y < height; y++) {
                    var value = greyscale.get(x, y);
                    value = (value - m)/s;
                    greyscale.set(x, y, value);
                }
            }
            return callback(null, greyscale);
        });
    } else {
        width = parseFloat(widthOrImage) || 1280;
        height = parseFloat(heightArg) || 720;
        callback(null, common.generate(width, height, function (x, y) {
            return 0;
        }));
    }
}

if (!inputWav) {
    var noise = common.generateNoise(phase, function (distance) {
        var logDistance = Math.log(distance/2);
        return Math.exp(-logDistance*logDistance);
    });
    var pattern = threshhold(noise, 0);
    common.writePng(pattern, 'noise.png', function (error) {
        if (error) throw error;
    });
    return;
}

var baseFreq = 40;
//var threshholdAmp = 0, threshholdAmpDarkness = 0.001;
//var threshholdAmp = -0.25, threshholdAmpDarkness = 1;
//var threshholdAmpFactor = 0, threshholdDarknessFactor = 0;
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

var backgroundStrength = 0;
var backgroundAmpStrength = 0.05;

getBackgroundImage(function (error, background) {
    if (error) throw error;

    // Phase setup
    var phaseStart = common.generate(width, height, function (x, y) {
        return Math.random()*2*Math.PI;
    });
    var phaseRates = common.generate(width, height, function (x, y) {
        var t = Math.random() - Math.random();
        return t*5;
    });

    ndarrayWav.open(inputWav, function (error, chunks) {
        if (error) throw error;
        var format = chunks.fmt;
        var sampleRate = format.sampleRate;
        var waveform = chunks.data;

        var waveformStd = common.std(waveform);

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
                var extract = common.generate(2, fftSize, function (channel, index) {
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

            var ampFunction = getSpectrum(0.1);

            var extractAmp2 = 0, extractAmpStep = 0.1;
            for (var dist = 0; dist < Math.sqrt(width*height); dist += extractAmpStep) {
                extractAmp2 += ampFunction(dist)*extractAmpStep;
            }
            var extractAmp = Math.sqrt(extractAmp2)/waveformStd;

            var phase = common.generate(width, height, function (x, y) {
                return phaseStart.get(x, y) + frameTime*phaseRates.get(x, y);
            });
            var noise = common.generateNoise(phase, function (distance) {
                return ampFunction(distance);
            });
            noise = common.generate(width, height, function (x, y) {
                return noise.get(x, y) + background.get(x, y)*(backgroundStrength + extractAmp*backgroundAmpStrength);
            });
            var pattern = common.threshholdProportion(noise, proportionCurve(extractAmp));
            //var pattern = threshhold(noise, threshholdAmp + extractAmp*threshholdAmpFactor, threshholdAmpDarkness + extractAmp*threshholdDarknessFactor);
            common.writePng(pattern, frameFile, function (error) {
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
        nextFrame();
    });
});
