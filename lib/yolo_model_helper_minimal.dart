import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart';
import 'package:flutter/services.dart' show rootBundle;

/// Minimal YOLO helper (≈140 lines) assuming:
/// - Single input tensor shape [1,H,W,3] float32 (NHWC)
/// - Single output tensor shape [1, num_preds, 4+num_classes] OR [1,4+num_classes,num_preds]
/// - Model produces raw (cx,cy,w,h) normalized to 0..1 (if >2 we auto-scale heuristically)
/// - You supply an already decoded `Image` (e.g. from file / memory)
///
/// Provides: init(), runOnImage(Image), dispose().
class YoloModelMinimal {
  final String modelPath;
  final String labelsPath;
  final int inputWidth;
  final int inputHeight;
  final double confThreshold;
  final double iouThreshold;

  Interpreter? _interpreter;
  List<String> _labels = const [];
  bool _ready = false;

  YoloModelMinimal({
    required this.modelPath,
    required this.labelsPath,
    required this.inputWidth,
    required this.inputHeight,
    this.confThreshold = 0.4,
    this.iouThreshold = 0.45,
  });

  bool get isReady => _ready;
  List<String> get labels => _labels;
  int get numClasses => _labels.length;

  Future<void> init({int threads = 4}) async {
    if (_ready) return;
    _labels = (await rootBundle.loadString(labelsPath))
        .split(RegExp(r'\r?\n'))
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList(growable: false);

    final options = InterpreterOptions()..threads = threads;
    _interpreter = await Interpreter.fromAsset(modelPath, options: options);

    // Ensure input tensor shape matches; resize if needed.
    final inputShape = _interpreter!.getInputTensor(0).shape;
    if (inputShape.length == 4 && (inputShape[1] != inputHeight || inputShape[2] != inputWidth)) {
      _interpreter!.resizeInputTensor(0, [1, inputHeight, inputWidth, 3]);
      _interpreter!.allocateTensors();
    }
    _ready = true;
  }

  void dispose() {
    _interpreter?.close();
    _ready = false;
  }

  /// Run detection on a decoded image (image package) and return detections.
  List<YoloDetection> runOnImage(Image img) {
    assert(_ready, 'Model not initialized');

    // 1. Preprocess (resize + normalize 0..1)
    final resized = copyResize(img, width: inputWidth, height: inputHeight);
    final input = List.generate(
      inputHeight,
      (y) => List.generate(
        inputWidth,
        (x) {
          final p = resized.getPixel(x, y);
            return [
              p.r.toDouble() / 255.0,
              p.g.toDouble() / 255.0,
              p.b.toDouble() / 255.0,
            ];
        },
        growable: false,
      ),
      growable: false,
    );
    final batched = [input]; // [1,H,W,3]

    // 2. Prepare output holder (dynamic shape: inspect tensor 0 after alloc)
    final outTensor = _interpreter!.getOutputTensor(0);
    final oShape = outTensor.shape; // Expect [1,N,4+C] or [1,4+C,N]
    final output = [ // we keep as list-of-lists-of-doubles
      List.generate(oShape[1], (_) => List<double>.filled(oShape[2], 0.0, growable: false), growable: false)
    ];

    try {
      _interpreter!.runForMultipleInputs([batched], {0: output}); // use runForMultipleInputs
    } catch (_) {
      // If shape mismatch (channels-first variant) retry transposed allocation
      if (oShape.length == 3 && oShape[2] < oShape[1]) {
        // attempt alt layout container
      }
    }

    // 3. Reformat output to [channels][num_preds]
    List<List<double>> channels;
    if (oShape.length == 3 && oShape[0] == 1) {
      final d1 = oShape[1];
      final d2 = oShape[2];
      final expected = 4 + numClasses;
      if (d2 == expected) {
        // layout [1,N,4+C]
        channels = List.generate(expected, (c) => List<double>.filled(d1, 0.0, growable: false), growable: false);
        for (int i = 0; i < d1; i++) {
          for (int c = 0; c < expected; c++) {
            channels[c][i] = output[0][i][c];
          }
        }
      } else if (d1 == expected) {
        // layout [1,4+C,N]
        channels = List.generate(expected, (c) => List<double>.from(output[0][c], growable: false), growable: false);
      } else {
        throw StateError('Unexpected output shape: $oShape expectedChannels=$expected');
      }
    } else {
      throw StateError('Unsupported output tensor shape $oShape');
    }

    // 4. Postprocess
    return _postprocess(channels, img.width, img.height);
  }

  List<YoloDetection> _postprocess(List<List<double>> chans, int origW, int origH) {
    final expected = 4 + numClasses;
    assert(chans.length == expected, 'Channel mismatch: ${chans.length} vs $expected');
    final numPreds = chans[0].length;

    // Detect if logits (need sigmoid)
    bool needSigmoid = false;
    for (int c = 4; c < expected && !needSigmoid; c++) {
      for (int i = 0; i < min(8, numPreds); i++) {
        final v = chans[c][i];
        if (v < -0.5 || v > 1.5) { needSigmoid = true; break; }
      }
    }
    double sig(double x) => 1.0 / (1.0 + exp(-x));

    final List<YoloDetection> candidates = [];
    for (int i = 0; i < numPreds; i++) {
      // (cx,cy,w,h)
      double cx = chans[0][i];
      double cy = chans[1][i];
      double w = chans[2][i];
      double h = chans[3][i];

      // Heuristic normalization if values look like pixel units
      if (w > 2 || h > 2 || cx > 2 || cy > 2) {
        double maxDim = max(max(cx.abs(), cy.abs()), max(w, h));
        double base = 640.0;
        if (maxDim > 1000) base = 1280.0; else if (maxDim < 300) base = 320.0;
        cx /= base; cy /= base; w /= base; h /= base;
      }

      double bestScore = 0.0; int bestCls = -1;
      for (int c = 0; c < numClasses; c++) {
        double v = chans[4 + c][i];
        if (needSigmoid) v = sig(v);
        if (v > bestScore) { bestScore = v; bestCls = c; }
      }
      if (bestCls >= 0 && bestScore >= confThreshold) {
        candidates.add(YoloDetection(bestCls, bestScore, cx, cy, w, h));
      }
    }

    // NMS
    candidates.sort((a,b)=> b.score.compareTo(a.score));
    final kept = <YoloDetection>[];
    for (final det in candidates) {
      bool keep = true;
      for (final k in kept) {
        if (_iou(det, k) > iouThreshold && det.classIndex == k.classIndex) { keep = false; break; }
      }
      if (keep) kept.add(det);
    }

    // Scale to original image size (cx,cy,w,h -> pixels)
    for (final d in kept) {
      d.cx *= origW; d.cy *= origH; d.w *= origW; d.h *= origH;
    }
    return kept;
  }

  double _iou(YoloDetection a, YoloDetection b) {
    final ax1 = a.cx - a.w/2, ay1 = a.cy - a.h/2, ax2 = a.cx + a.w/2, ay2 = a.cy + a.h/2;
    final bx1 = b.cx - b.w/2, by1 = b.cy - b.h/2, bx2 = b.cx + b.w/2, by2 = b.cy + b.h/2;
    final x1 = max(ax1, bx1), y1 = max(ay1, by1), x2 = min(ax2, bx2), y2 = min(ay2, by2);
    final iw = (x2 - x1); final ih = (y2 - y1);
    if (iw <= 0 || ih <= 0) return 0.0;
    final inter = iw * ih;
    final areaA = (ax2-ax1) * (ay2-ay1);
    final areaB = (bx2-bx1) * (by2-by1);
    final union = areaA + areaB - inter;
    if (union <= 0) return 0.0;
    return inter / union;
  }
}

class YoloDetection {
  final int classIndex;
  final double score;
  double cx; // center x (pixels after scaling)
  double cy; // center y
  double w;  // width
  double h;  // height
  YoloDetection(this.classIndex, this.score, this.cx, this.cy, this.w, this.h);

  @override
  String toString() => 'YoloDetection(cls=$classIndex score=${score.toStringAsFixed(2)} cx=${cx.toStringAsFixed(1)} cy=${cy.toStringAsFixed(1)} w=${w.toStringAsFixed(1)} h=${h.toStringAsFixed(1)})';
}
