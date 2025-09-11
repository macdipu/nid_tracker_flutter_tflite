import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'bbox.dart';
import 'yolo_model_helper.dart';

class ImageDetectPage extends StatefulWidget {
  final YoloModelHelper model;
  const ImageDetectPage({super.key, required this.model});

  @override
  State<ImageDetectPage> createState() => _ImageDetectPageState();
}

class _ImageDetectPageState extends State<ImageDetectPage> {
  static const double maxImageWidgetHeight = 400;
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<int> _classes = [];
  List<List<double>> _bboxes = [];
  List<double> _scores = [];
  List<List<double>>? _rawOutput;
  int? _imgW;
  int? _imgH;

  double _confTh = 0.4;
  double _iouTh = 0.1;

  List<Color> _colors = [];

  @override
  void initState() {
    super.initState();
    _colors = List<Color>.generate(
      widget.model.numClasses,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withAlpha(255),
    );
  }

  void _ensureColor(int cls) {
    if (cls >= _colors.length) {
      final extra = cls - _colors.length + 1;
      _colors.addAll(List<Color>.generate(
        extra,
        (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withAlpha(255),
      ));
    }
  }

  Future<void> _pickImage() async {
    final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return;
    _imgW = decoded.width;
    _imgH = decoded.height;
    final preds = widget.model.infer(decoded); // sync (already in isolate version retained)
    setState(() {
      _imageFile = File(file.path);
      _rawOutput = preds;
    });
    _runPost();
  }

  void _runPost() {
    if (_rawOutput == null || _imgW == null || _imgH == null) return;
    final (c, b, s) = widget.model.postprocess(
      _rawOutput!,
      _imgW!,
      _imgH!,
      confidenceThreshold: _confTh,
      iouThreshold: _iouTh,
    );
    setState(() {
      _classes = c;
      _bboxes = b;
      _scores = s;
    });
  }

  @override
  Widget build(BuildContext context) {
    double resizeFactor = 1;
    final displayWidth = MediaQuery.of(context).size.width;
    if (_imgW != null && _imgH != null) {
      final k1 = displayWidth / _imgW!;
      final k2 = maxImageWidgetHeight / _imgH!;
      resizeFactor = min(k1, k2);
    }

    final bboxWidgets = <Widget>[];
    for (int i = 0; i < _bboxes.length; i++) {
      final box = _bboxes[i];
      final cls = _classes[i];
      _ensureColor(cls);
      final label = (cls >= 0 && cls < widget.model.labels.length) ? widget.model.labels[cls] : 'cls$cls';
      bboxWidgets.add(
        Bbox(
          box[0] * resizeFactor,
          box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            label,
            _scores[i],
            _colors[cls],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Gallery Detection')),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImage,
        child: const Icon(Icons.photo_library_outlined),
      ),
      body: ListView(
        children: [
          SizedBox(
            height: maxImageWidgetHeight,
            child: Center(
              child: Stack(
                children: [
                  if (_imageFile == null)
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.image_outlined, size: 80),
                        Text('Pick an image', style: Theme.of(context).textTheme.headlineSmall),
                      ],
                    )
                  else
                    Image.file(_imageFile!),
                  ...bboxWidgets,
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _slider(
            label: 'Confidence',
            value: _confTh,
            onChanged: (v) => setState(() { _confTh = v; _runPost(); }),
          ),
          _slider(
            label: 'IoU',
            value: _iouTh,
            onChanged: (v) => setState(() { _iouTh = v; _runPost(); }),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _slider({required String label, required double value, required ValueChanged<double> onChanged}) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('$label: ${(value * 100).toStringAsFixed(0)}%'),
            ],
          ),
          Slider(
            value: value,
            min: 0,
            max: 1,
            divisions: 100,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}

