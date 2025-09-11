import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:camera/camera.dart';

class YoloModelHelper {
  final String modelPath;
  final String labelsPath;
  final int inWidth;
  final int inHeight;
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  List<String> labels = const [];
  late List<int> _inputShape;
  bool _channelsFirst = false;
  late Object _inputType; // changed from TfLiteType to Object
  bool _initialized = false;
  bool _disposed = false;
  bool _inputIsFloat = true; // new
  int frameCounter = 10;
  List<List<List<double>>>? _bufNHWC; // shape [H][W][3]
  List<List<List<double>>>? _bufNCHW; // shape [3][H][W]

  YoloModelHelper({
    required this.modelPath,
    required this.labelsPath,
    required this.inWidth,
    required this.inHeight,
  });

  int get numClasses => labels.length;
  bool get isInitialized => _initialized;

  Future<void> init() async {
    if (_initialized) return;
    try {
      final raw = await rootBundle.loadString(labelsPath);
      labels = raw
          .split(RegExp(r'\r?\n'))
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList(growable: false);

      final options = InterpreterOptions()..threads = 4; // enable multithreading
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _inputShape = _interpreter!.getInputTensor(0).shape;
      _inputType = _interpreter!.getInputTensor(0).type;
      final typeStr = _inputType.toString().toLowerCase();
      _inputIsFloat = typeStr.contains('float');

      if (_inputShape.length != 4) {
        throw StateError('Unsupported input tensor shape: $_inputShape');
      }

      if (_inputShape[1] == 3) {
        _channelsFirst = true; // NCHW
        if (_inputShape[2] != inHeight || _inputShape[3] != inWidth) {
          _interpreter!.resizeInputTensor(0, [1, 3, inHeight, inWidth]);
        }
      } else {
        _channelsFirst = false; // NHWC
        if (_inputShape[1] != inHeight || _inputShape[2] != inWidth || _inputShape[3] != 3) {
          _interpreter!.resizeInputTensor(0, [1, inHeight, inWidth, 3]);
        }
      }
      _interpreter!.allocateTensors();
      _inputShape = _interpreter!.getInputTensor(0).shape;

      // Wrap with isolate for async inference
      _isolateInterpreter = await IsolateInterpreter.create(address: _interpreter!.address);
      _initialized = true;

      if (kDebugMode) {
        final outTensor = _interpreter!.getOutputTensor(0);
        debugPrint('✅ YOLO init done: Input=$_inputShape layout=${_channelsFirst ? 'NCHW' : 'NHWC'} type=$_inputType float=$_inputIsFloat');
        debugPrint('✅ Output shape: ${outTensor.shape}');
        debugPrint('✅ Labels: ${labels.length}');
      }
    } catch (e, st) {
      debugPrint('❌ Failed to initialize model: $e');
      debugPrint('Stack trace:\n$st');
      rethrow;
    }
  }

  // Legacy sync inference (kept for compatibility) - still runs in main isolate.
  List<List<double>> infer(Image image) {
    assert(_interpreter != null, 'Model not initialized');
    final input = _preprocessImageToNestedList(image);
    return _runOnInputSync(input);
  }

  Future<List<List<double>>> inferImageAsync(Image image) async {
    assert(_isolateInterpreter != null, 'Model not initialized');
    final input = _preprocessImageToNestedList(image);
    return _runOnInputAsync(input);
  }

  Future<(List<int>, List<List<double>>, List<double>)> inferAndPostprocessImageAsync(
    Image image, {
    double confidenceThreshold = 0.7,
    double iouThreshold = 0.1,
    bool agnostic = false,
  }) async {
    final preds = await inferImageAsync(image);
    return postprocess(
      preds,
      image.width,
      image.height,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnostic,
    );
  }

  // Camera image streaming inference. Converts YUV420 -> RGB -> model input (nearest resize inline)
  Future<(List<int>, List<List<double>>, List<double>)?> inferCameraImage(
    CameraImage cameraImage, {
    double confidenceThreshold = 0.5,
    double iouThreshold = 0.4,
    bool agnostic = false,
    int processEveryNFrames = 1,
  }) async {
    if (!_initialized) return null;
    frameCounter++;
    if (frameCounter % processEveryNFrames != 0) return null; // skip frames for speed
    final Stopwatch sw = Stopwatch()..start();
    final dynamic input = _preprocessCameraImage(cameraImage);
    final preds = await _runOnInputAsync(input);
    final (classes, bboxes, scores) = postprocess(
      preds,
      cameraImage.width,
      cameraImage.height,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnostic,
    );
    if (kDebugMode) {
      debugPrint('📷 Frame infer ${sw.elapsedMilliseconds}ms det=${bboxes.length}');
      if (bboxes.isEmpty) {
        // print simple score stats
        double maxScore = 0;
        if (preds.length > 4) {
          for (int j = 4; j < preds.length; j++) {
            for (int i = 0; i < preds[j].length; i++) {
              if (preds[j][i] > maxScore) maxScore = preds[j][i];
            }
          }
        }
        debugPrint('🔍 Max raw class score (pre-threshold) = $maxScore');
      }
    }
    return (classes, bboxes, scores);
  }

  void dispose() {
    if (_disposed) return;
    try {
      _isolateInterpreter?.close();
      _interpreter?.close();
    } catch (_) {}
    _disposed = true;
  }

  // --- Internal helpers ---
  dynamic _preprocessImageToNestedList(Image image) {
    final resized = copyResize(image, width: inWidth, height: inHeight);
    if (_channelsFirst) {
      final input = [
        [
          List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)),
          List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)),
          List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)),
        ]
      ];
      for (int y = 0; y < inHeight; y++) {
        for (int x = 0; x < inWidth; x++) {
          final p = resized.getPixel(x, y);
          input[0][0][y][x] = p.rNormalized.toDouble();
          input[0][1][y][x] = p.gNormalized.toDouble();
          input[0][2][y][x] = p.bNormalized.toDouble();
        }
      }
      return input;
    } else {
      return [
        List.generate(
          inHeight,
          (y) => List.generate(
            inWidth,
            (x) {
              final p = resized.getPixel(x, y);
              return [
                p.rNormalized.toDouble(),
                p.gNormalized.toDouble(),
                p.bNormalized.toDouble(),
              ];
            },
            growable: false,
          ),
          growable: false,
        )
      ];
    }
  }

  dynamic _preprocessCameraImage(CameraImage image) {
    // Assumes YUV420.
    final int srcW = image.width;
    final int srcH = image.height;
    final planeY = image.planes[0];
    final planeU = image.planes.length > 1 ? image.planes[1] : null;
    final planeV = image.planes.length > 2 ? image.planes[2] : null;
    final int yRowStride = planeY.bytesPerRow;
    final int uvRowStride = planeU?.bytesPerRow ?? 0;
    final int uvPixelStride = planeU?.bytesPerPixel ?? 1; // Often 2

    double scaleY = srcH / inHeight;
    double scaleX = srcW / inWidth;
    double norm(double v) => _inputIsFloat ? (v / 255.0) : v;

    if (_channelsFirst) {
      _bufNCHW ??= [
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0)), // R
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0)), // G
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0)), // B
      ];
      final cbuf = _bufNCHW!; // [3][H][W]
      for (int y = 0; y < inHeight; y++) {
        final srcY = (y * scaleY).clamp(0, srcH - 1).toInt();
        for (int x = 0; x < inWidth; x++) {
          final srcX = (x * scaleX).clamp(0, srcW - 1).toInt();
          final yIndex = srcY * yRowStride + srcX;
          final int Y = planeY.bytes[yIndex];
          int U = 128, V = 128;
          if (planeU != null && planeV != null) {
            final uvY = (srcY >> 1);
            final uvX = (srcX >> 1);
            final uIndex = uvY * uvRowStride + uvX * uvPixelStride;
            final vIndex = uvY * (planeV.bytesPerRow) + uvX * (planeV.bytesPerPixel ?? uvPixelStride);
            U = planeU.bytes[uIndex];
            V = planeV.bytes[vIndex];
          }
          final rgb = _yuvToRgb(Y, U, V); // values 0..1
          cbuf[0][y][x] = norm(rgb[0] * 255.0);
          cbuf[1][y][x] = norm(rgb[1] * 255.0);
          cbuf[2][y][x] = norm(rgb[2] * 255.0);
        }
      }
      return [cbuf]; // add batch dim
    } else {
      _bufNHWC ??= List.generate(
        inHeight,
        (_) => List.generate(inWidth, (_) => List<double>.filled(3, 0.0)),
      ); // [H][W][3]
      final hbuf = _bufNHWC!;
      for (int y = 0; y < inHeight; y++) {
        final srcY = (y * scaleY).clamp(0, srcH - 1).toInt();
        for (int x = 0; x < inWidth; x++) {
          final srcX = (x * scaleX).clamp(0, srcW - 1).toInt();
          final yIndex = srcY * yRowStride + srcX;
          final int Y = planeY.bytes[yIndex];
          int U = 128, V = 128;
          if (planeU != null && planeV != null) {
            final uvY = (srcY >> 1);
            final uvX = (srcX >> 1);
            final uIndex = uvY * uvRowStride + uvX * uvPixelStride;
            final vIndex = uvY * (planeV.bytesPerRow) + uvX * (planeV.bytesPerPixel ?? uvPixelStride);
            U = planeU.bytes[uIndex];
            V = planeV.bytes[vIndex];
          }
          final rgb = _yuvToRgb(Y, U, V);
          final p = hbuf[y][x];
          p[0] = norm(rgb[0] * 255.0);
          p[1] = norm(rgb[1] * 255.0);
          p[2] = norm(rgb[2] * 255.0);
        }
      }
      return [hbuf]; // add batch dim
    }
  }

  List<double> _yuvToRgb(int y, int u, int v) {
    double yf = y.toDouble();
    double uf = (u - 128).toDouble();
    double vf = (v - 128).toDouble();
    double r = (yf + 1.402 * vf).clamp(0, 255) / 255.0;
    double g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255) / 255.0;
    double b = (yf + 1.772 * uf).clamp(0, 255) / 255.0;
    return [r, g, b];
  }

  List<List<double>> _runOnInputSync(dynamic input) {
    final outTensor = _interpreter!.getOutputTensor(0);
    final oShape = outTensor.shape;
    final int d1 = oShape[1];
    final int d2 = oShape[2];
    final int expectedChannels = 4 + numClasses;
    final output = [List.generate(d1, (_) => List<double>.filled(d2, 0.0, growable: false), growable: false)];
    _interpreter!.run(input, output);
    return _reformatOutput(output, oShape, expectedChannels);
  }

  Future<List<List<double>>> _runOnInputAsync(dynamic input) async {
    final outTensor = _interpreter!.getOutputTensor(0);
    final oShape = outTensor.shape;
    final int d1 = oShape[1];
    final int d2 = oShape[2];
    final int expectedChannels = 4 + numClasses;
    final output = [List.generate(d1, (_) => List<double>.filled(d2, 0.0, growable: false), growable: false)];
    await _isolateInterpreter!.run(input, output);
    return _reformatOutput(output, oShape, expectedChannels);
  }

  List<List<double>> _reformatOutput(dynamic output, List<int> oShape, int expectedChannels) {
    if (oShape.length != 3 || oShape[0] != 1) {
      throw StateError('Unsupported output shape: $oShape');
    }
    final int d1 = oShape[1];
    final int d2 = oShape[2];
    if (d1 == expectedChannels) {
      return List.generate(d1, (c) => List<double>.from(output[0][c]));
    } else if (d2 == expectedChannels) {
      final channelsFirst = List.generate(expectedChannels, (c) => List<double>.filled(d1, 0.0, growable: false));
      for (int i = 0; i < d1; i++) {
        for (int c = 0; c < expectedChannels; c++) {
          channelsFirst[c][i] = output[0][i][c];
        }
      }
      return channelsFirst;
    } else {
      throw StateError('Cannot match output channels. Shape=$oShape expectedChannels=$expectedChannels (labels=${labels.length})');
    }
  }

  (List<int>, List<List<double>>, List<double>) postprocess(
    List<List<double>> unfilteredBboxes,
    int imageWidth,
    int imageHeight, {
    double confidenceThreshold = 0.7,
    double iouThreshold = 0.1,
    bool agnostic = false,
  }) {
    try {
      final rawCopy = [for (var c in unfilteredBboxes) List<double>.from(c)]; // keep original for fallback scaling
      final (classes, bboxes, scores) = postProcessDetections(
        unfilteredBboxes,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
      );
      int outOfBounds = 0;
      for (var bbox in bboxes) {
        bbox[0] *= imageWidth;
        bbox[1] *= imageHeight;
        bbox[2] *= imageWidth;
        bbox[3] *= imageHeight;
        if (bbox[0] < 0 || bbox[0] > imageWidth * 1.5 || bbox[1] < 0 || bbox[1] > imageHeight * 1.5) {
          outOfBounds++;
        }
      }
      if (bboxes.isNotEmpty && outOfBounds / bboxes.length > 0.7) {
        // Fallback: assume original coords already in pixel units of model input size
        if (kDebugMode) debugPrint('⚠️ Falling back to pixel-space scaling heuristic');
        final (cls2, b2, sc2) = postProcessDetections(
          rawCopy,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold,
        );
        for (var bbox in b2) {
          // scale from model input size to image size
            bbox[0] *= (imageWidth / inWidth);
            bbox[1] *= (imageHeight / inHeight);
            bbox[2] *= (imageWidth / inWidth);
            bbox[3] *= (imageHeight / inHeight);
        }
        return (cls2, b2, sc2);
      }
      return (classes, bboxes, scores);
    } catch (e, st) {
      debugPrint('❌ Postprocessing failed: $e');
      debugPrint('Stack trace:\n$st');
      rethrow;
    }
  }
}

(List<int>, List<List<double>>, List<double>) postProcessDetections(
  List<List<double>> rawOutput, {
  double confidenceThreshold = 0.7,
  double iouThreshold = 0.4,
}) {
  if (rawOutput.length < 5) {
    return (<int>[], <List<double>>[], <double>[]);
  }
  final int numChannels = rawOutput.length;
  final int numClasses = numChannels - 4;
  if (numClasses <= 0) {
    return (<int>[], <List<double>>[], <double>[]);
  }
  final int numPredictions = rawOutput[0].length;
  bool needSigmoid = false;
  for (int j = 4; j < 4 + numClasses && !needSigmoid; j++) {
    for (int i = 0; i < min(numPredictions, 32); i++) {
      final v = rawOutput[j][i];
      if (v > 1.5 || v < 0) { needSigmoid = true; break; }
    }
  }
  double _sig(double x) => 1.0 / (1.0 + exp(-x));
  List<int> bestClasses = [];
  List<double> bestScores = [];
  List<int> boxesToSave = [];
  for (int i = 0; i < numPredictions; i++) {
    double bestScore = 0;
    int bestCls = -1;
    for (int j = 4; j < 4 + numClasses; j++) {
      double clsScore = rawOutput[j][i];
      if (needSigmoid) clsScore = _sig(clsScore);
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestCls = j - 4;
      }
    }
    if (bestScore > confidenceThreshold && bestCls >= 0) {
      bestClasses.add(bestCls);
      bestScores.add(bestScore);
      boxesToSave.add(i);
    }
  }
  List<List<double>> candidateBoxes = [];
  for (var index in boxesToSave) {
    var cx = rawOutput[0][index];
    var cy = rawOutput[1][index];
    var w = rawOutput[2][index];
    var h = rawOutput[3][index];
    // If coordinates look like pixels already (>1) normalize by max dimension heuristic (assume trained on inWidth/inHeight typical 640)
    if (w > 2 || h > 2 || cx > 2 || cy > 2) {
      // convert to relative (assuming training size 640) fallback
      const double base = 640.0;
      cx /= base; cy /= base; w /= base; h /= base;
    }
    candidateBoxes.add([cx, cy, w, h]);
  }
  if (candidateBoxes.isEmpty) {
    return (<int>[], <List<double>>[], <double>[]);
  }
  List<int> argSortList = List.generate(bestScores.length, (i) => i);
  argSortList.sort((a, b) => -bestScores[a].compareTo(bestScores[b]));
  List<int> sortedBestClasses = [for (var i in argSortList) bestClasses[i]];
  List<List<double>> sortedCandidateBoxes = [for (var i in argSortList) candidateBoxes[i]];
  List<double> sortedScores = [for (var i in argSortList) bestScores[i]];
  List<List<double>> finalBboxes = [];
  List<double> finalScores = [];
  List<int> finalClasses = [];
  while (sortedCandidateBoxes.isNotEmpty) {
    var bbox1xywh = sortedCandidateBoxes.removeAt(0);
    finalBboxes.add(bbox1xywh);
    var bbox1xyxy = xywh2xyxy(bbox1xywh);
    finalScores.add(sortedScores.removeAt(0));
    var class1 = sortedBestClasses.removeAt(0);
    finalClasses.add(class1);
    List<int> indexesToRemove = [];
    for (int i = 0; i < sortedCandidateBoxes.length; i++) {
      if ((class1 == sortedBestClasses[i]) &&
          computeIou(bbox1xyxy, xywh2xyxy(sortedCandidateBoxes[i])) > iouThreshold) {
        indexesToRemove.add(i);
      }
    }
    for (var index in indexesToRemove.reversed) {
      sortedCandidateBoxes.removeAt(index);
      sortedBestClasses.removeAt(index);
      sortedScores.removeAt(index);
    }
  }
  return (finalClasses, finalBboxes, finalScores);
}

List<double> xywh2xyxy(List<double> bbox) {
  double halfWidth = bbox[2] / 2;
  double halfHeight = bbox[3] / 2;
  return [
    bbox[0] - halfWidth,
    bbox[1] - halfHeight,
    bbox[0] + halfWidth,
    bbox[1] + halfHeight,
  ];
}

double computeIou(List<double> bbox1, List<double> bbox2) {
  assert(bbox1[0] < bbox1[2]);
  assert(bbox1[1] < bbox1[3]);
  assert(bbox2[0] < bbox2[2]);
  assert(bbox2[1] < bbox2[3]);
  double xLeft = max(bbox1[0], bbox2[0]);
  double yTop = max(bbox1[1], bbox2[1]);
  double xRight = min(bbox1[2], bbox2[2]);
  double yBottom = min(bbox1[3], bbox2[3]);
  if (xRight < xLeft || yBottom < yTop) {
    return 0;
  }
  double intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  double bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
  double bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
  double iou = intersectionArea / (bbox1Area + bbox2Area - intersectionArea);
  if (iou < 0) return 0;
  if (iou > 1) return 1;
  return iou;
}
