var inputs = [];
for (var i = 2; i < process.argv.length; i += 2) {
    inputs.push({
        directory: process.argv[i],
        colour: (process.argv[i + 1] || '').split(',').map(parseFloat)
    });
}
var outputDir = inputs.pop().directory;

var path = require('path');
var async = require('async');
var fse = require('fs-extra');
var getPixels = require('get-pixels');
var savePixels = require('save-pixels');

var common = require('./common');

async.map(inputs, function (input, callback) {
    fse.readdir(input.directory, function (error, list) {
        if (error) return callback(error);
        var images = {};
        list.forEach(function (item) {
            if (item[0] === '.') return;
            if (!/\.(png|jpe?g)$/i.test(item)) return;
            images[item] = true;
        });
        callback(null, images);
    });
}, function (error, listings) {
    if (error) throw error;
    var commonFiles = null;
    listings.forEach(function (images) {
        if (!commonFiles) return commonFiles = images;
        Object.keys(commonFiles).forEach(function (key) {
            if (!images[key]) {
                delete commonFiles[key];
            }
        });
    });
    var images = Object.keys(commonFiles);
    images.sort();
    async.mapSeries(images, function (image, callback) {
        async.map(inputs, function (input, callback) {
            getPixels(path.join(input.directory, image), callback);
        }, function (error, images) {
            var firstImage = images[0];
            var width = firstImage.shape[0];
            var height = firstImage.shape[1];
            var channels = firstImage.shape[2];
            for (var x = 0; x < width; x++) {
                for (var y = 0; y < height; y++) {
                    for (var c = 0; c < 3; c++) {
                        var value = 255;
                        images.forEach(function (pixels, imageIndex) {
                            var colourValue = inputs[imageIndex].colour[c]/255;
                            var imageValue = (pixels.get(x, y, c) || 0)/255;
                            value *= (colourValue*(1 - imageValue) + imageValue);
                        });
                        firstImage.set(x, y, c, value);
                    }
                    firstImage.set(x, y, 3, 255);
                }
            }
            common.writePng(firstImage, path.join(outputDir, image), function (error) {
                if (error) return callback(error);
                process.stdout.clearLine();
                process.stdout.write('\r' + image);
                callback(null);
            });
        });
    }, function (error) {
        if (error) throw error;
        console.log('\ndone');
    })
});
