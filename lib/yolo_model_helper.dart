import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart';
import 'package:flutter/services.dart' show rootBundle;

class YoloModelHelper {
  final String modelPath;
  final String labelsPath;
  final int inWidth;
  final int inHeight;
  Interpreter? _interpreter;
  List<String> labels = const [];
  late List<int> _inputShape;
  bool _channelsFirst = false;

  YoloModelHelper({
    required this.modelPath,
    required this.labelsPath,
    required this.inWidth,
    required this.inHeight,
  });

  int get numClasses => labels.length;
  bool get isInitialized => _interpreter != null;

  Future<void> init() async {
    try {
      // Load labels dynamically
      final raw = await rootBundle.loadString(labelsPath);
      labels = raw
          .split(RegExp(r'\r?\n'))
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList(growable: false);

      _interpreter = await Interpreter.fromAsset(modelPath);
      _inputShape = _interpreter!.getInputTensor(0).shape; // e.g. [1,640,640,3] or [1,3,640,640]

      if (_inputShape.length != 4) {
        throw StateError('Unsupported input tensor shape: $_inputShape');
      }

      // Determine layout
      if (_inputShape[1] == 3) {
        // [1,3,H,W]
        _channelsFirst = true;
        if (_inputShape[2] != inHeight || _inputShape[3] != inWidth) {
          if (kDebugMode) {
            debugPrint('🔧 Resizing NCHW input tensor from $_inputShape to [1,3,$inHeight,$inWidth]');
          }
          _interpreter!.resizeInputTensor(0, [1, 3, inHeight, inWidth]);
        }
      } else {
        // Assume NHWC [1,H,W,3]
        _channelsFirst = false;
        if (_inputShape[1] != inHeight || _inputShape[2] != inWidth || _inputShape[3] != 3) {
          if (kDebugMode) {
            debugPrint('🔧 Resizing NHWC input tensor from $_inputShape to [1,$inHeight,$inWidth,3]');
          }
          _interpreter!.resizeInputTensor(0, [1, inHeight, inWidth, 3]);
        }
      }

      _interpreter!.allocateTensors();
      _inputShape = _interpreter!.getInputTensor(0).shape; // refresh

      if (kDebugMode) {
        final outTensor = _interpreter!.getOutputTensor(0);
        debugPrint('✅ Input shape: $_inputShape layout: ${_channelsFirst ? 'NCHW' : 'NHWC'}');
        debugPrint('✅ Output tensor shape: ${outTensor.shape}');
        debugPrint('✅ Loaded ${labels.length} labels');
      }
    } catch (e, st) {
      debugPrint('❌ Failed to initialize model: $e');
      debugPrint('Stack trace:\n$st');
      rethrow;
    }
  }

  List<List<double>> infer(Image image) {
    assert(_interpreter != null, 'The model must be initialized');
    try {
      // Resize & (optionally) pad letterbox if needed (simple resize here)
      final resized = copyResize(image, width: inWidth, height: inHeight);

      dynamic input; // structure must match interpreter input shape
      if (_channelsFirst) {
        // Shape [1,3,H,W]
        input = [
          [
            List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)), // R temp, will overwrite
            List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)), // G
            List.generate(inHeight, (y) => List<double>.filled(inWidth, 0.0, growable: false)), // B
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
      } else {
        // Shape [1,H,W,3]
        input = [
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

      // Prepare output buffer
      final outTensor = _interpreter!.getOutputTensor(0);
      final oShape = outTensor.shape; // Expect [1,N,4+nc] or [1,4+nc,N]
      if (oShape.length != 3 || oShape[0] != 1) {
        throw StateError('Unsupported output shape: $oShape');
      }
      final int d1 = oShape[1];
      final int d2 = oShape[2];
      final int expectedChannels = 4 + numClasses;

      final output = [
        List.generate(
          d1,
          (_) => List<double>.filled(d2, 0.0, growable: false),
          growable: false,
        )
      ];

      final t0 = DateTime.now().millisecondsSinceEpoch;
      _interpreter!.run(input, output);
      final dt = DateTime.now().millisecondsSinceEpoch - t0;
      if (kDebugMode) debugPrint('⏱ Inference: ${dt}ms');

      // Reformat to channels-first [4+nc, numPreds]
      List<List<double>> channelsFirst;
      if (d1 == expectedChannels) {
        channelsFirst = List.generate(d1, (c) => List<double>.from(output[0][c]));
      } else if (d2 == expectedChannels) {
        channelsFirst = List.generate(expectedChannels, (c) => List<double>.filled(d1, 0.0, growable: false));
        for (int i = 0; i < d1; i++) {
          for (int c = 0; c < expectedChannels; c++) {
            channelsFirst[c][i] = output[0][i][c];
          }
        }
      } else {
        throw StateError('Cannot match output channels. Shape=$oShape expectedChannels=$expectedChannels (labels=${labels.length})');
      }
      return channelsFirst;
    } catch (e, st) {
      debugPrint('❌ Inference failed: $e');
      debugPrint('Stack trace:\n$st');
      rethrow;
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
      final (classes, bboxes, scores) = postProcessDetections(
        unfilteredBboxes,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
      );

      for (var bbox in bboxes) {
        bbox[0] *= imageWidth;
        bbox[1] *= imageHeight;
        bbox[2] *= imageWidth;
        bbox[3] *= imageHeight;
      }
      return (classes, bboxes, scores);
    } catch (e, st) {
      debugPrint('❌ Postprocessing failed: $e');
      debugPrint('Stack trace:\n$st');
      rethrow;
    }
  }

  (List<int>, List<List<double>>, List<double>) inferAndPostprocess(
    Image image, {
    double confidenceThreshold = 0.7,
    double iouThreshold = 0.1,
    bool agnostic = false,
  }) {
    final preds = infer(image);
    return postprocess(
      preds,
      image.width,
      image.height,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnostic,
    );
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

  List<int> bestClasses = [];
  List<double> bestScores = [];
  List<int> boxesToSave = [];

  for (int i = 0; i < numPredictions; i++) {
    double bestScore = 0;
    int bestCls = -1;
    for (int j = 4; j < 4 + numClasses; j++) {
      double clsScore = rawOutput[j][i];
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
    candidateBoxes.add([
      rawOutput[0][index],
      rawOutput[1][index],
      rawOutput[2][index],
      rawOutput[3][index],
    ]);
  }

  if (candidateBoxes.isEmpty) {
    return (<int>[], <List<double>>[], <double>[]);
  }

  var sortedBestScores = List<double>.from(bestScores);
  // Stable sort by preserving indices when equal scores
  List<int> argSortList = List.generate(sortedBestScores.length, (i) => i);
  argSortList.sort((a, b) => -bestScores[a].compareTo(bestScores[b]));

  List<int> sortedBestClasses = [for (var i in argSortList) bestClasses[i]];
  List<List<double>> sortedCandidateBoxes = [
    for (var i in argSortList) candidateBoxes[i]
  ];
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
          computeIou(bbox1xyxy, xywh2xyxy(sortedCandidateBoxes[i])) >
              iouThreshold) {
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
