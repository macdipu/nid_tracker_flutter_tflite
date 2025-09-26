import 'package:flutter/material.dart';
import 'package:nid_tracker_flutter_tflite/yolo_model_helper.dart';
import 'live_detect_page_minimal.dart' show LiveDetectPage;

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late final YoloModelHelper model;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    model = YoloModelHelper(
      modelPath: 'assets/models/yolov11n.tflite',
      labelsPath: 'assets/models/labels.txt',
      inputWidth: 640,
      inputHeight: 640,
    );
    _init();
  }

  Future<void> _init() async {
    try {
      await model.init();
      if (mounted) setState(() => _loading = false);
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Demo')),
      body: Center(
        child: _loading
            ? const CircularProgressIndicator()
            : _error != null
                ? Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text('Init error: $_error'),
                  )
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const SizedBox(height: 24),
                      ElevatedButton.icon(
                        icon: const Icon(Icons.videocam_outlined),
                        label: const Text('Live Camera Detection Minimal'),
                        onPressed: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => LiveDetectPage(model: model),
                          ),
                        ),
                      ),
                    ],
                  ),
      ),
    );
  }
}
