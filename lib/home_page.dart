import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:nid_tracker_flutter_tflite/yolo_model_helper.dart';
import 'bbox.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const double maxImageWidgetHeight = 400;

  final YoloModelHelper model = YoloModelHelper(
    modelPath : 'assets/models/yolov11n.tflite',
    labelsPath :'assets/models/labels.txt',
    inHeight: 640,
    inWidth:  640,
  );
  File? imageFile;

  double confidenceThreshold = 0.4;
  double iouThreshold = 0.1;
  bool agnosticNMS = false;

  List<List<double>>? inferenceOutput;
  List<int> classes = [];
  List<List<double>> bboxes = [];
  List<double> scores = [];

  int? imageWidth;
  int? imageHeight;

  // Dynamic bbox colors
  List<Color> bboxColors = [];

  @override
  void initState() {
    super.initState();
    model.init().then((_) {
      setState(() {
        bboxColors = List<Color>.generate(
          model.numClasses,
          (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withAlpha(255),
        );
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    // Ensure we have at least some colors if detections exceed prepared list
    void ensureColorCapacity(int cls) {
      if (cls >= bboxColors.length) {
        final extra = cls - bboxColors.length + 1;
        bboxColors.addAll(List<Color>.generate(
          extra,
          (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withAlpha(255),
        ));
      }
    }

    final ImagePicker picker = ImagePicker();

    final double displayWidth = MediaQuery.of(context).size.width;

    const textPadding = EdgeInsets.symmetric(horizontal: 16);

    double resizeFactor = 1;

    if (imageWidth != null && imageHeight != null) {
      double k1 = displayWidth / imageWidth!;
      double k2 = maxImageWidgetHeight / imageHeight!;
      resizeFactor = min(k1, k2);
    }

    List<Bbox> bboxesWidgets = [];
    for (int i = 0; i < bboxes.length; i++) {
      final box = bboxes[i];
      final boxClass = classes[i];
      ensureColorCapacity(boxClass);
      final labelText = (boxClass >= 0 && boxClass < model.labels.length)
          ? model.labels[boxClass]
          : 'cls$boxClass';
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labelText,
            scores[i],
            bboxColors[boxClass]),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO')),
      body: ListView(
        children: [
          InkWell(
            onTap: () async {
              final XFile? newImageFile =
                  await picker.pickImage(source: ImageSource.gallery);
              if (newImageFile != null) {
                setState(() {
                  imageFile = File(newImageFile.path);
                });
                final image =
                    img.decodeImage(await newImageFile.readAsBytes())!;
                imageWidth = image.width;
                imageHeight = image.height;
                inferenceOutput = model.infer(image);
                updatePostprocess();
              }
            },
            child: SizedBox(
              height: maxImageWidgetHeight,
              child: Center(
                child: Stack(
                  children: [
                    if (imageFile == null)
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(
                            Icons.file_open_outlined,
                            size: 80,
                          ),
                          Text(
                            'Pick an image',
                            style: Theme.of(context).textTheme.headlineMedium,
                          ),
                        ],
                      )
                    else
                      Image.file(imageFile!),
                    ...bboxesWidgets,
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 30),
          Padding(
            padding: textPadding,
            child: Row(
              children: [
                Text(
                  'Confidence threshold:',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(width: 8),
                Text(
                  '${(confidenceThreshold * 100).toStringAsFixed(0)}%',
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          const Padding(
            padding: textPadding,
            child: Text(
              'If high, only the clearly recognizable objects will be detected. If low even not clear objects will be detected.',
            ),
          ),
          Slider(
            value: confidenceThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            onChanged: (value) {
              setState(() {
                confidenceThreshold = value;
                updatePostprocess();
              });
            },
          ),
          const SizedBox(height: 8),
          Padding(
            padding: textPadding,
            child: Row(
              children: [
                Text(
                  'IoU threshold',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(width: 8),
                Text(
                  '${(iouThreshold * 100).toStringAsFixed(0)}%',
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          const Padding(
            padding: textPadding,
            child: Text(
              'If high, overlapped objects will be detected. If low, only separated objects will be correctly detected.',
            ),
          ),
          Slider(
            value: iouThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            onChanged: (value) {
              setState(() {
                iouThreshold = value;
                updatePostprocess();
              });
            },
          ),
        ],
      ),
    );
  }

  void updatePostprocess() {
    if (inferenceOutput == null) {
      return;
    }
    final (newClasses, newBboxes, newScores) = model.postprocess(
      inferenceOutput!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );
    debugPrint('Detected ${newBboxes.length} bboxes');
    setState(() {
      classes = newClasses;
      bboxes = newBboxes;
      scores = newScores;
    });
  }
}
