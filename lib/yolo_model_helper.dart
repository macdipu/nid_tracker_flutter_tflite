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
  late Object _inputType;
  bool _initialized = false;
  bool _disposed = false;
  bool _inputIsFloat = true;
  int frameCounter = 0; // Changed to start at 0
  List<List<List<double>>>? _bufNHWC;
  List<List<List<double>>>? _bufNCHW;

  // Performance optimization: reuse output buffer
  List<List<double>>? _outputBuffer;

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

      // Optimize interpreter options for better performance
      final options = InterpreterOptions()
        ..threads = 4; // Use 4 threads

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

      // Pre-allocate output buffer for better performance
      final outTensor = _interpreter!.getOutputTensor(0);
      final oShape = outTensor.shape;
      if (oShape.length == 3 && oShape[0] == 1) {
        final d1 = oShape[1];
        final d2 = oShape[2];
        _outputBuffer = List.generate(d1, (_) => List<double>.filled(d2, 0.0, growable: false), growable: false);
      }

      // Wrap with isolate for async inference
      _isolateInterpreter = await IsolateInterpreter.create(address: _interpreter!.address);
      _initialized = true;

      if (kDebugMode) {
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

  // Legacy sync inference (kept for compatibility)
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
        double confidenceThreshold = 0.5, // Lowered default threshold
        double iouThreshold = 0.4,
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

  // Optimized camera image inference with better YUV handling
  Future<(List<int>, List<List<double>>, List<double>)?> inferCameraImage(
      CameraImage cameraImage, {
        double confidenceThreshold = 0.3, // Lowered for better detection
        double iouThreshold = 0.4,
        bool agnostic = false,
        int processEveryNFrames = 3,
        bool tryRotationFallback = true, // run a 90°-rotated pass too
        int preferClassIndex = -1, // prefer results containing this class
      }) async {
    if (!_initialized) return null;

    frameCounter++;
    if (frameCounter % processEveryNFrames != 0) return null;

    final Stopwatch sw = Stopwatch()..start();

    try {
      // 0° pass
      final dynamic input0 = _preprocessCameraImageOptimized(cameraImage);
      final preds0 = await _runOnInputAsync(input0);
      var (classes0, bboxes0, scores0) = postprocess(
        preds0,
        cameraImage.width,
        cameraImage.height,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
        agnostic: agnostic,
      );

      if (!tryRotationFallback) {
        if (kDebugMode && sw.elapsedMilliseconds > 100) {
          debugPrint('📷 Slow frame: ${sw.elapsedMilliseconds}ms det=${bboxes0.length}');
        }
        return (classes0, bboxes0, scores0);
      }

      // 90° clockwise rotated pass (reuse preprocessed buffer, rotate tensor)
      (List<int>, List<List<double>>, List<double>)? rotatedRes;
      try {
        final rotatedInput = _rotatePreprocessedInput90(input0, clockwise: true);
        final preds90 = await _runOnInputAsync(rotatedInput);
        var (cls90, b90, sc90) = postprocess(
          preds90,
          cameraImage.height, // swapped because input is rotated
          cameraImage.width,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold,
          agnostic: agnostic,
        );
        // map 90° results back to original camera orientation
        b90 = _mapRotatedBboxesBack(b90, cameraImage.width, cameraImage.height, clockwise: true);
        rotatedRes = (cls90, b90, sc90);
      } catch (e) {
        // if rotation path fails for any reason, ignore it
        rotatedRes = null;
      }

      double scoreOf(List<int> cls, List<double> sc) {
        double s = 0.0;
        for (final v in sc) s += v;
        s += cls.length * 0.05; // slight preference to more boxes
        if (preferClassIndex >= 0 && cls.contains(preferClassIndex)) s += 1.0;
        return s;
      }

      final baseScore = scoreOf(classes0, scores0);
      final rotScore = rotatedRes != null ? scoreOf(rotatedRes.$1, rotatedRes.$3) : double.negativeInfinity;

      final useRotated =
          rotatedRes != null &&
          ( // prefer presence of target class
            (preferClassIndex >= 0 &&
             rotatedRes!.$1.contains(preferClassIndex) &&
             !classes0.contains(preferClassIndex)) ||
            // otherwise pick higher aggregate score
            (rotScore > baseScore)
          );

      if (kDebugMode && sw.elapsedMilliseconds > 100) {
        debugPrint('📷 Slow frame: ${sw.elapsedMilliseconds}ms det0=${bboxes0.length} det90=${rotatedRes?.$2.length ?? 0} chosen=${useRotated ? "90°" : "0°"}');
      }

      if (useRotated) {
        return rotatedRes!;
      } else {
        return (classes0, bboxes0, scores0);
      }
    } catch (e) {
      debugPrint('❌ Camera inference error: $e');
      return null;
    }
  }

  void dispose() {
    if (_disposed) return;
    try {
      _isolateInterpreter?.close();
      _interpreter?.close();
    } catch (_) {}
    _disposed = true;
  }

  // --- Optimized Internal helpers ---
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

  // Optimized camera preprocessing with better YUV420 handling
  dynamic _preprocessCameraImageOptimized(CameraImage image) {
    final int srcW = image.width;
    final int srcH = image.height;
    final planeY = image.planes[0];
    final planeU = image.planes.length > 1 ? image.planes[1] : null;
    final planeV = image.planes.length > 2 ? image.planes[2] : null;

    final int yRowStride = planeY.bytesPerRow;
    final int uvRowStride = planeU?.bytesPerRow ?? 0;
    final int uvPixelStride = planeU?.bytesPerPixel ?? 1;

    // Use more precise scaling
    final double scaleX = srcW / inWidth;
    final double scaleY = srcH / inHeight;

    double norm(double v) => _inputIsFloat ? (v / 255.0) : v;

    if (_channelsFirst) {
      // Reuse buffer if available
      _bufNCHW ??= [
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0, growable: false)), // R
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0, growable: false)), // G
        List.generate(inHeight, (_) => List<double>.filled(inWidth, 0.0, growable: false)), // B
      ];
      final cbuf = _bufNCHW!; // [3][H][W]

      for (int y = 0; y < inHeight; y++) {
        final srcYf = y * scaleY;
        final srcY = srcYf.floor().clamp(0, srcH - 1);
        final srcYNext = (srcYf.ceil()).clamp(0, srcH - 1);
        final yFrac = srcYf - srcY;

        for (int x = 0; x < inWidth; x++) {
          final srcXf = x * scaleX;
          final srcX = srcXf.floor().clamp(0, srcW - 1);
          final srcXNext = (srcXf.ceil()).clamp(0, srcW - 1);
          final xFrac = srcXf - srcX;

          // Bilinear interpolation for Y channel
          final y1Index = srcY * yRowStride + srcX;
          final y2Index = srcY * yRowStride + srcXNext;
          final y3Index = srcYNext * yRowStride + srcX;
          final y4Index = srcYNext * yRowStride + srcXNext;

          final Y1 = planeY.bytes[y1Index].toDouble();
          final Y2 = planeY.bytes[y2Index].toDouble();
          final Y3 = planeY.bytes[y3Index].toDouble();
          final Y4 = planeY.bytes[y4Index].toDouble();

          final Y = Y1 * (1 - xFrac) * (1 - yFrac) +
              Y2 * xFrac * (1 - yFrac) +
              Y3 * (1 - xFrac) * yFrac +
              Y4 * xFrac * yFrac;

          // Get UV values (subsampled)
          int U = 128, V = 128;
          if (planeU != null && planeV != null) {
            final uvY = (srcY >> 1).clamp(0, (srcH >> 1) - 1);
            final uvX = (srcX >> 1).clamp(0, (srcW >> 1) - 1);
            final uIndex = uvY * uvRowStride + uvX * uvPixelStride;
            final vIndex = uvY * (planeV.bytesPerRow) + uvX * (planeV.bytesPerPixel ?? uvPixelStride);

            if (uIndex < planeU.bytes.length && vIndex < planeV.bytes.length) {
              U = planeU.bytes[uIndex];
              V = planeV.bytes[vIndex];
            }
          }

          final rgb = _yuvToRgbOptimized(Y.toInt(), U, V);
          cbuf[0][y][x] = norm(rgb[0] * 255.0);
          cbuf[1][y][x] = norm(rgb[1] * 255.0);
          cbuf[2][y][x] = norm(rgb[2] * 255.0);
        }
      }
      return [cbuf]; // add batch dim
    } else {
      // NHWC format - reuse buffer
      _bufNHWC ??= List.generate(
        inHeight,
            (_) => List.generate(inWidth, (_) => List<double>.filled(3, 0.0, growable: false), growable: false),
      );
      final hbuf = _bufNHWC!;

      for (int y = 0; y < inHeight; y++) {
        final srcYf = y * scaleY;
        final srcY = srcYf.floor().clamp(0, srcH - 1);

        for (int x = 0; x < inWidth; x++) {
          final srcXf = x * scaleX;
          final srcX = srcXf.floor().clamp(0, srcW - 1);

          final yIndex = srcY * yRowStride + srcX;
          final int Y = planeY.bytes[yIndex];

          int U = 128, V = 128;
          if (planeU != null && planeV != null) {
            final uvY = (srcY >> 1).clamp(0, (srcH >> 1) - 1);
            final uvX = (srcX >> 1).clamp(0, (srcW >> 1) - 1);
            final uIndex = uvY * uvRowStride + uvX * uvPixelStride;
            final vIndex = uvY * (planeV.bytesPerRow) + uvX * (planeV.bytesPerPixel ?? uvPixelStride);

            if (uIndex < planeU.bytes.length && vIndex < planeV.bytes.length) {
              U = planeU.bytes[uIndex];
              V = planeV.bytes[vIndex];
            }
          }

          final rgb = _yuvToRgbOptimized(Y, U, V);
          final p = hbuf[y][x];
          p[0] = norm(rgb[0] * 255.0);
          p[1] = norm(rgb[1] * 255.0);
          p[2] = norm(rgb[2] * 255.0);
        }
      }
      return [hbuf]; // add batch dim
    }
  }

  // Optimized YUV to RGB conversion with better color space handling
  List<double> _yuvToRgbOptimized(int y, int u, int v) {
    // ITU-R BT.601 conversion with proper clamping
    final double yf = y.toDouble();
    final double uf = (u - 128).toDouble();
    final double vf = (v - 128).toDouble();

    final double r = (yf + 1.402 * vf).clamp(0, 255) / 255.0;
    final double g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255) / 255.0;
    final double b = (yf + 1.772 * uf).clamp(0, 255) / 255.0;

    return [r, g, b];
  }

  // Rotate the preprocessed input tensor by 90 degrees.
  // Works for both NHWC [1,H,W,3] and NCHW [1,3,H,W].
  // If input width != height, returns the original input without rotation.
  dynamic _rotatePreprocessedInput90(dynamic input, {bool clockwise = true}) {
    if (inWidth != inHeight) {
      // rotation would swap dimensions; our interpreter is square (typical for YOLO),
      // so in non-square cases skip rotation fallback.
      return input;
    }

    if (_channelsFirst) {
      // input: [1][3][H][W]
      final src = input[0] as List; // [3][H][W]
      final H = inHeight;
      final W = inWidth;

      final dst = [
        [
          List.generate(H, (_) => List<double>.filled(W, 0.0, growable: false)),
          List.generate(H, (_) => List<double>.filled(W, 0.0, growable: false)),
          List.generate(H, (_) => List<double>.filled(W, 0.0, growable: false)),
        ]
      ];

      for (int c = 0; c < 3; c++) {
        for (int y = 0; y < H; y++) {
          for (int x = 0; x < W; x++) {
            final v = src[c][y][x] as double;
            int ny, nx;
            if (clockwise) {
              ny = x;
              nx = W - 1 - y;
            } else {
              ny = H - 1 - x;
              nx = y;
            }
            dst[0][c][ny][nx] = v;
          }
        }
      }
      return dst;
    } else {
      // input: [1][H][W][3]
      final src = input[0] as List; // [H][W][3]
      final H = inHeight;
      final W = inWidth;

      final dst = [
        List.generate(H, (_) => List.generate(W, (_) => List<double>.filled(3, 0.0, growable: false), growable: false), growable: false)
      ];

      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          final p = src[y][x] as List; // [3]
          int ny, nx;
          if (clockwise) {
            ny = x;
            nx = W - 1 - y;
          } else {
            ny = H - 1 - x;
            nx = y;
          }
          final outP = dst[0][ny][nx] as List<double>;
          outP[0] = p[0] as double;
          outP[1] = p[1] as double;
          outP[2] = p[2] as double;
        }
      }
      return dst;
    }
  }

  // Map bboxes (cx,cy,w,h in pixels) predicted on a 90°-rotated frame
  // back to the original camera orientation.
  List<List<double>> _mapRotatedBboxesBack(List<List<double>> boxes, int rawW, int rawH, {bool clockwise = true}) {
    final out = <List<double>>[];
    for (final b in boxes) {
      if (b.length < 4) continue;
      final cx = b[0];
      final cy = b[1];
      final w  = b[2];
      final h  = b[3];

      double nx, ny, nw, nh;
      if (clockwise) {
        nx = rawW - cy;
        ny = cx;
      } else {
        nx = cy;
        ny = rawH - cx;
      }
      nw = h;
      nh = w;
      out.add([nx, ny, nw, nh]);
    }
    return out;
  }

  List<List<double>> _runOnInputSync(dynamic input) {
    final output = _outputBuffer != null ? [_outputBuffer!] : [
      List.generate(_interpreter!.getOutputTensor(0).shape[1],
              (_) => List<double>.filled(_interpreter!.getOutputTensor(0).shape[2], 0.0, growable: false),
          growable: false)
    ];

    // Updated to use runForMultipleInputs for consistency/future multi-input extensibility
    final outputs = <int, Object>{0: output};
    _interpreter!.runForMultipleInputs([input], outputs);
    final oShape = _interpreter!.getOutputTensor(0).shape;
    final expectedChannels = 4 + numClasses;
    return _reformatOutput(output, oShape, expectedChannels);
  }

  Future<List<List<double>>> _runOnInputAsync(dynamic input) async {
    final output = _outputBuffer != null ? [_outputBuffer!] : [
      List.generate(_interpreter!.getOutputTensor(0).shape[1],
              (_) => List<double>.filled(_interpreter!.getOutputTensor(0).shape[2], 0.0, growable: false),
          growable: false)
    ];

    await _isolateInterpreter!.run(input, output);
    final oShape = _interpreter!.getOutputTensor(0).shape;
    final expectedChannels = 4 + numClasses;
    return _reformatOutput(output, oShape, expectedChannels);
  }

  List<List<double>> _reformatOutput(dynamic output, List<int> oShape, int expectedChannels) {
    if (oShape.length != 3 || oShape[0] != 1) {
      throw StateError('Unsupported output shape: $oShape');
    }
    final int d1 = oShape[1];
    final int d2 = oShape[2];

    if (d1 == expectedChannels) {
      return List.generate(d1, (c) => List<double>.from(output[0][c]), growable: false);
    } else if (d2 == expectedChannels) {
      final channelsFirst = List.generate(expectedChannels,
              (c) => List<double>.filled(d1, 0.0, growable: false), growable: false);
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
        double confidenceThreshold = 0.3, // Lowered default
        double iouThreshold = 0.4,
        bool agnostic = false,
      }) {
    try {
      final rawCopy = [for (var c in unfilteredBboxes) List<double>.from(c)];
      final (classes, bboxes, scores) = postProcessDetections(
        unfilteredBboxes,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
      );

      // Enhanced coordinate handling for better orientation support
      int outOfBounds = 0;
      for (var bbox in bboxes) {
        // Store original values for debugging
        final origCx = bbox[0];
        final origCy = bbox[1];

        bbox[0] *= imageWidth;
        bbox[1] *= imageHeight;
        bbox[2] *= imageWidth;
        bbox[3] *= imageHeight;

        // More lenient bounds checking
        if (bbox[0] < -imageWidth * 0.2 || bbox[0] > imageWidth * 1.2 ||
            bbox[1] < -imageHeight * 0.2 || bbox[1] > imageHeight * 1.2) {
          outOfBounds++;
        }
      }

      // Adjusted fallback threshold
      if (bboxes.isNotEmpty && outOfBounds / bboxes.length > 0.5) {
        if (kDebugMode) debugPrint('⚠️ Falling back to pixel-space scaling heuristic');
        final (cls2, b2, sc2) = postProcessDetections(
          rawCopy,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold,
        );
        for (var bbox in b2) {
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

// Optimized post-processing with better detection handling
(List<int>, List<List<double>>, List<double>) postProcessDetections(
    List<List<double>> rawOutput, {
      double confidenceThreshold = 0.3, // Lowered default
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

  // Optimized sigmoid detection - check fewer samples for speed
  bool needSigmoid = false;
  for (int j = 4; j < min(4 + numClasses, rawOutput.length) && !needSigmoid; j++) {
    for (int i = 0; i < min(numPredictions, 16); i++) { // Reduced sample size
      final v = rawOutput[j][i];
      if (v > 1.5 || v < -0.5) {
        needSigmoid = true;
        break;
      }
    }
  }

  double _sig(double x) => 1.0 / (1.0 + exp(-x));

  List<int> bestClasses = [];
  List<double> bestScores = [];
  List<int> boxesToSave = [];

  // Pre-allocate lists for better performance
  bestClasses = List<int>.filled(0, 0, growable: true);
  bestScores = List<double>.filled(0, 0.0, growable: true);
  boxesToSave = List<int>.filled(0, 0, growable: true);

  for (int i = 0; i < numPredictions; i++) {
    double bestScore = 0;
    int bestCls = -1;

    for (int j = 4; j < 4 + numClasses; j++) {
      if (j >= rawOutput.length) break;

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

  if (boxesToSave.isEmpty) {
    return (<int>[], <List<double>>[], <double>[]);
  }

  List<List<double>> candidateBoxes = [];
  for (var index in boxesToSave) {
    var cx = rawOutput[0][index];
    var cy = rawOutput[1][index];
    var w = rawOutput[2][index];
    var h = rawOutput[3][index];

    // Enhanced coordinate normalization with better heuristics
    if (w > 2 || h > 2 || cx > 2 || cy > 2) {
      // Try to detect the training resolution automatically
      double maxDim = max(max(cx.abs(), cy.abs()), max(w, h));
      double base = 640.0; // Default assumption

      if (maxDim > 1000) {
        base = 1280.0; // Likely trained on higher resolution
      } else if (maxDim < 300) {
        base = 320.0; // Likely trained on lower resolution
      }

      cx /= base;
      cy /= base;
      w /= base;
      h /= base;
    }

    candidateBoxes.add([cx, cy, w, h]);
  }

  // Optimized NMS with pre-sorted scores
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

    // Remove in reverse order to maintain indices
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
  // Add bounds checking for robustness
  if (bbox1.length < 4 || bbox2.length < 4) return 0.0;
  if (bbox1[0] >= bbox1[2] || bbox1[1] >= bbox1[3]) return 0.0;
  if (bbox2[0] >= bbox2[2] || bbox2[1] >= bbox2[3]) return 0.0;

  double xLeft = max(bbox1[0], bbox2[0]);
  double yTop = max(bbox1[1], bbox2[1]);
  double xRight = min(bbox1[2], bbox2[2]);
  double yBottom = min(bbox1[3], bbox2[3]);

  if (xRight <= xLeft || yBottom <= yTop) {
    return 0.0;
  }

  double intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  double bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
  double bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

  if (bbox1Area <= 0 || bbox2Area <= 0) return 0.0;

  double unionArea = bbox1Area + bbox2Area - intersectionArea;
  if (unionArea <= 0) return 0.0;

  double iou = intersectionArea / unionArea;
  return iou.clamp(0.0, 1.0);
}