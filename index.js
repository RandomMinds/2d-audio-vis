var fse = require('fs-extra');
var path = require('path');
var crypto = require('crypto');

var async = require('async');
var jstat = require('jstat').jStat;
var ndarray = require('ndarray');
var ndarrayWav = require('ndarray-wav');
var ndarrayFft = require('ndarray-fft');
var ndarraySavePixels = require('save-pixels');

function tanh(x) {
	if (x >= 20) return 1;
	if (x <= -20) return -1;
	return Math.tanh(x);
}

function Perlin(seed, options) {
	var thisPerlin = this;
	this._seed = seed;
	this._options = options || {};
	
	var width = this._width = options.width || 320;
	var height = this._height = options.height || Math.round(width/4*3/16)*16;
	
	var freqBase = this._freqBase = options.freqBase || 40;
	var phaseBase = this._phaseBase = this._create(width, height, function (x, y) {
		var dx = (x > width/2) ? x - width : x;
		var dy = (y > height/2) ? y - height : y;
		var random = thisPerlin._random(dx + '_' + dy + '_phaseBase');
		return random*Math.PI*2;
	});
	var phaseRate = this._phaseRate = this._create(width, height, function (x, y) {
		var dx = (x > width/2) ? x - width : x;
		var dy = (y > height/2) ? y - height : y;
		var random = thisPerlin._random(dx + '_' + dy + '_phaseRate');
		return (random - 0.5)*7;
	});
}
Perlin.prototype = {
	_random: function (extra) {
		var hash = crypto.createHash('sha256');
		hash.update(this._seed);
		hash.update(extra);
		var buffer = hash.digest();
		return buffer.readUInt32BE(0)/4294967296;
	},
	_create: function (width, height, valueFn) {
		width = width || this._width;
		height = height || this._height;
		var array = new Float64Array(width*height);
		var nd = ndarray(array, [width, height]);
		for (var x = 0; valueFn && x < width; x++) {
			for (var y = 0; y < height; y++) {
				nd.set(x, y, valueFn(x, y));
			}
		}
		return nd;
	},
	noise: function (time, ampFn, density) {
		var width = this._width, height = this._height;
		if (density <= 0 || density >= 1) return this._create(width, height, function (x, y) {
			return density > 0 ? 1 : 0;
		});

		var freqBase = this._freqBase;
		var phaseBase = this._phaseBase, phaseRate = this._phaseRate;
		var spectrumReal = this._create(), spectrumImag = this._create();
		for (var x = 0; x < width; x++) {
			for (var y = 0; y < height; y++) {
				var dx = (x > width/2) ? x - width : x;
				var dy = (y > height/2) ? y - height : y;
				var dist = Math.sqrt(dx*dx + dy*dy);
				var freq = freqBase*dist;
				var amp = ampFn(freq);
				var phase = phaseBase.get(x, y) + time*phaseRate.get(x, y);
				spectrumReal.set(x, y, amp*Math.cos(phase));
				spectrumImag.set(x, y, amp*Math.sin(phase));
			}
		}
		for (var x = 0; x < width; x++) {
			ndarrayFft(-1, spectrumReal.pick(x), spectrumImag.pick(x));
		}
		for (var y = 0; y < height; y++) {
			ndarrayFft(-1, spectrumReal.pick(null, y), spectrumImag.pick(null, y));
		}
		var sum2 = 0;
		for (var x = 0; x < width; x++) {
			for (var y = 0; y < height; y++) {
				var value = spectrumReal.get(x, y);
				sum2 += value*value;
			}
		}
		var std = (sum2 > 0) ? Math.sqrt(sum2/width/height) : 1;
		
		var threshhold = jstat.normal.inv(density, 0, std);
		var fuzzyWidth = (this._options.fuzzy || 0)*std;
		return this._create(width, height, function (x, y) {
			var relative = spectrumReal.get(x, y) + threshhold;
			if (!fuzzyWidth) return (relative > 0) ? 1 : 0;
			var value = tanh(relative/fuzzyWidth)*0.5 + 0.5;
			return value;
		});
	}
};

function rgbStack(stack, options) {
	startrgb = options.rgb || [255, 255, 255];
	var combineFunction = function (current, target, amount) {
		// Multiplicative
		var factor = 1 + (target - 255)/255*amount;
		return current*factor;
	};
	if (options.combine === 'light') {
		var darkCombine = combineFunction;
		combineFunction = function (current, target, amount) {
			return 255 - darkCombine(255 - current, 255 - target, amount);
		};
	}
	
	var width = stack[0].noise.shape[0], height = stack[0].noise.shape[1];
	var array = new Array(3*width*height);
	var rgb = ndarray(array, [width, height, 3]);
	for (var x = 0; x < width; x++) {
		for (var y = 0; y < height; y++) {
			for (var c = 0; c < 3; c++) {
				var value = startrgb[c];
				stack.forEach(function (layer) {
					var amount = layer.noise.get(x, y);
					value = combineFunction(value, layer.rgb[c], amount);
				});
				rgb.set(x, y, c, Math.round(value));
			}
		}
	}
	return rgb;
}

function createVisualiser(wavFile, perlin, options, callback) {
	var lowFreq = options.lowFreq || 100;
	function freqMultiplier(freq) {
		if (freq < lowFreq) return freq/lowFreq;
		return 1;
	}
	function densityMap(rms) {
		return rms*Math.sqrt(2);
		return Math.min(1, rms*2);
	}

	ndarrayWav.open(wavFile, function (error, chunks) {
		if (error) return callback(error);
		var format = chunks.fmt;
		var waveform = chunks.data;
		var windowDuration = 0.2, windowPadding = 2, windowBias = 3;
		var windowDurationSamples = windowDuration*format.sampleRate;
		var windowSamples = windowDuration*windowPadding*format.sampleRate;
		windowSamples = Math.pow(2, Math.ceil(Math.log2(windowDurationSamples)));
		console.log('\tloaded: ' + path.basename(wavFile));
		var windowOffset = -windowDuration*Math.pow(0.5, 1/windowBias); // Middle of the window
		
		var extractL = ndarray(new Float64Array(windowSamples));
		var extractR = ndarray(new Float64Array(windowSamples));
		var rChannel = waveform.shape[0] > 1 ? 1 : 0;
		
		var vis = {
			duration: waveform.shape[1]/format.sampleRate,
			noise: function (time) {
				var sample = (time + windowOffset)*format.sampleRate;
				var start = Math.round(sample), end = Math.round(sample + windowDurationSamples);
				var sum2 = 0, valueL, valueR;
				for (var i = 0; i < windowSamples; i++) {
					extractL.set(i, 0);
					extractR.set(i, 0);
				}
				var windowSum2 = 0;
				for (var i = start; i < end; i++) {
					var ratio = (i - start + 0.5)/(end - start);
					ratio = Math.pow(ratio, windowBias);
					var windowValue = 0.5 - 0.5*Math.cos(ratio*Math.PI*2);
					windowSum2 += windowValue*windowValue;
					if (i >= 0 && i < waveform.shape[1]) {
						extractL.set(i - start, valueL = waveform.get(0, i)*windowValue);
						extractR.set(i - start, valueR = waveform.get(rChannel, i)*windowValue);
						sum2 += valueL*valueL + valueR*valueR;
					}
				}
				var rms = Math.sqrt(sum2/2/windowSum2);
				ndarrayFft(1, extractL, extractR);
				function spectrum(f) {
					var freqRatio = f/format.sampleRate;
					if (freqRatio <= 0 || freqRatio >= 0.5) return 0;
					
					var index1 = Math.round(freqRatio*windowSamples);
					var index2 = windowSamples - index1;
					var leftR = (extractL.get(index1) + extractL.get(index2))/2;
					var leftI = (extractR.get(index1) - extractR.get(index2))/2;
					var rightR = (extractR.get(index1) + extractR.get(index2))/2;
					var rightI = (-extractL.get(index1) + extractL.get(index2))/2;
					var power = leftR*leftR + leftI*leftI + rightR*rightR + rightI*rightI;
					/*
					      var leftReal = (extractL.get(index) + extract.pick(0, index2))/2;
					      var leftImag = (extractR.get(index) - extract.pick(1, index2))/2;
					      var rightReal = (extract.pick(1, index) + extract.pick(1, index2))/2;
					      var rightImag = (-extract.pick(0, index) + extract.pick(0, index2))/2;
					*/
					var mag = Math.sqrt(power);
					return freqMultiplier(f)*mag;
				}
				return perlin.noise(time, spectrum, densityMap(rms));
			}
		};
		return callback(null, vis);
	});
}

if (require.main === module) {
	var updateLineTimeout, updateLineTimeoutMs = 30, nextUpdateText;
	function updateLine(text, sync) {
		nextUpdateText = text;
		function actualUpdate() {
			updateLineTimeout = null;
			var readline = require('readline');
			readline.clearLine(process.stdout);
			readline.cursorTo(process.stdout, 0);
			var maxLength = process.stdout.columns - 1;
			if (nextUpdateText.length > maxLength) {
				nextUpdateText = nextUpdateText.substring(0, maxLength - 3) + '...';
			}
			process.stdout.write('\r' + nextUpdateText);
		}
		if (sync) {
			actualUpdate();
		} else {
			updateLineTimeout = updateLineTimeout || setTimeout(actualUpdate, updateLineTimeoutMs);
		}
	}
	function logEstimate(frame, time, duration, startEncodeMs) {
		var ratio = time/duration;
		var endEstimateMs = (Date.now() - startEncodeMs)*(1 - ratio)/ratio;
		var endEstimateH = Math.floor(endEstimateMs/1000/60/60);
		var endEstimateM = Math.floor(endEstimateMs/1000/60)%60;
		var endEstimateS = Math.floor(endEstimateMs/1000)%60;
		var logLine = '\rframe ' + frame + ' (' + Math.round(time) + '/' + Math.round(duration) + ')\tremaining: ' + endEstimateH + 'h' + endEstimateM + 'm' + endEstimateS + 's';
		updateLine(logLine);
		if (typeof process.send === 'function') {
			process.send({
				estimateMs: endEstimateMs,
				logLine: logLine
			});
		}
	}

	var argIndex = 2;
	var configFile = process.argv[argIndex++];
	var relativeDir = process.cwd();
	var imageDir = process.argv[argIndex++];
	var width = parseFloat(process.argv[argIndex++] || 320), height = parseFloat(process.argv[argIndex++]);
	var frameRate = process.argv[argIndex++] || 30;
	var skipSpec = process.argv[argIndex++] || '';
	var skipParts = skipSpec.split('/').map(parseFloat);
	if (skipParts.length === 1 && skipParts[0] > 0) {
		// All right, let's fork
		var args = process.argv.slice(2);
		var numChildren = args.pop();
		var child_process = require('child_process');
		
		var latestMessages = [];
		for (var i = 0; i < numChildren; i++) {
			latestMessages.push({estimateMs: 0, logLine: 'Initialising'});
		}
		var workers = latestMessages.map(function (message, index) {
			var worker = child_process.fork('index.js', args.concat(i + '/' + numChildren), {silent: true, stdio: ['ignore', 'ignore', process.stderr]});
			worker.on('message', function (message) {
				latestMessages[index] = message;
				logLatest();
			});
			return worker;
		});
		updateLine('Spawned ' + workers.length + ' workers');
		function logLatest() {
			var latestMs = 0, logLine = '';
			latestMessages.forEach(function (message) {
				if (message.estimateMs >= latestMs) {
					latestMes = message.estimateMs;
					logLine = message.logLine;
				}
			});
			updateLine(logLine);
		}
		return;
	}
	
	if (!configFile || !imageDir) {
		console.error('Usage:\n\tnode index <input.wav|input.json> <folder/> <?width=320> <?height> <?framerate>\n');
		process.exit(1);
	}
	fse.ensureDirSync(imageDir);
	
	var config;
	if (/\.json/i.test(configFile)) {
		config = JSON.parse(fse.readFileSync(configFile));
		relativeDir = path.dirname(configFile);
	} else {
		config = {
			layers: [
				{
					wav: configFile,
					rgb: [0, 0, 0]
				}
			]
		};
	}

	var perlinSeed = 'seed';
	var sharedPerlin;
	var layerCount = 0;
	async.map(config.layers, function (spec, callback) {
		var layerIndex = layerCount++;
		var options = {width: width, height: height, fuzzy: 0};
		if ('fuzzy' in config) options.fuzzy = config.fuzzy;
		if ('fuzzy' in spec) options.fuzzy = spec.fuzzy;

		var seed = perlinSeed + (config.independent ? layerIndex : '');
		var wavFile = path.resolve(relativeDir, spec.wav);
		createVisualiser(wavFile, new Perlin(seed, options), options, callback);
	}, function (error, visList) {
		if (error) throw error;

		var duration = visList[0].duration;
		for (var i = 0; i < visList.length; i++) duration = Math.max(duration, visList[i].duration);

		var startEncodeMs = Date.now();
		function drawFrame(frameN) {
			var time = (frameN + 0.5)/frameRate;
			if (time >= duration) {
				updateLine('');
				return;
			}
			if (skipParts.length === 2 && !isNaN(skipParts[0]) && skipParts[1] > 0) {
				var modulo = frameN%skipParts[1];
				if (modulo != skipParts[0]%skipParts[1]) return drawFrame(frameN + 1);
			}

			logEstimate(frameN, time, duration, startEncodeMs);
			var stack = visList.map(function (vis, index) {
				return {
					rgb: config.layers[index].rgb || [0, 0, 0],
					noise: vis.noise(time)
				};
			});
			var image = rgbStack(stack, config);
		
			var imageFile = path.join(imageDir, 'frame' + ('000000' + frameN).substr(-5) + '.png');
			var imageStream = ndarraySavePixels(image, 'png', outputStream);
			var outputStream = fse.createWriteStream(imageFile);
			imageStream.pipe(outputStream);
			outputStream.on('error', function (error) {
				throw error;
			});
			outputStream.on('close', function () {
				drawFrame(frameN + 1);
			});
		}
		drawFrame(0);
	});
	/*
	var stack = [];
	stack.push({
		rgb: [64, 128, 192],
		noise: perlin.noise(0, function (f) {
			if (!f) return 0;
			return 1/f;
		}, 0.5)
	});
	stack.push({
		rgb: [192, 128, 64],
		noise: perlin.noise(1, function (f) {
			if (!f) return 0;
			return 1/f/f;
		}, 0.5)
	});
	stack.push({
		rgb: [0, 192, 0],
		noise: perlin.noise(2, function (f) {
			if (!f) return 0;
			return 1/f/Math.sqrt(f);
		}, 0.5)
	});
	*/
}