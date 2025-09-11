import 'dart:async';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'bbox.dart';
import 'yolo_model_helper.dart';

class LiveDetectPage extends StatefulWidget {
  final YoloModelHelper model;
  const LiveDetectPage({super.key, required this.model});
  @override
  State<LiveDetectPage> createState() => _LiveDetectPageState();
}

class _LiveDetectPageState extends State<LiveDetectPage> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _initializing = true;
  bool _streaming = false;
  bool _processing = false;

  List<int> _classes = [];
  List<List<double>> _bboxes = [];
  List<double> _scores = [];
  List<Color> _colors = [];

  double _confTh = 0.25; // visibility
  double _iouTh = 0.4;

  int? _frameW;
  int? _frameH;

  int _frames = 0;
  int _lastDetCount = 0;
  late DateTime _fpsStart;
  double _fps = 0;
  Timer? _fpsTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _fpsStart = DateTime.now();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        if (mounted) setState(() => _initializing = false);
        return;
      }
      final cam = cams.first;
      _controller = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _initializing = false);
      _startStream();
      _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _computeFps());
    } catch (e) {
      if (mounted) setState(() => _initializing = false);
      debugPrint('Camera init error: $e');
    }
  }

  void _startStream() {
    if (_controller == null || _streaming) return;
    _streaming = true;
    _controller!.startImageStream(_onNewFrame);
  }

  void _onNewFrame(CameraImage frame) {
    _frames++;
    _frameW = frame.width;
    _frameH = frame.height;
    if (_processing) return;
    _processing = true;
    widget.model
        .inferCameraImage(frame, confidenceThreshold: _confTh, iouThreshold: _iouTh)
        .then((res) {
      if (!mounted || res == null) return;
      final (classes, bboxes, scores) = res;
      if (classes.isNotEmpty) {
        _lastDetCount = classes.length;
      } else if (scores.isNotEmpty) {
        debugPrint('No detections (min=${scores.reduce(min)} max=${scores.reduce(max)})');
      }
      for (final c in classes) _ensureColor(c);
      setState(() {
        _classes = classes;
        _bboxes = bboxes;
        _scores = scores;
      });
    }).catchError((e) {
      debugPrint('Frame infer error: $e');
    }).whenComplete(() => _processing = false);
  }

  void _computeFps() {
    final now = DateTime.now();
    final elapsedMs = now.difference(_fpsStart).inMilliseconds;
    if (elapsedMs > 0) {
      _fps = (_frames * 1000) / elapsedMs;
      _fpsStart = now;
      _frames = 0;
      if (mounted) setState(() {});
    }
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

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!mounted || _controller == null) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.stopImageStream();
      _streaming = false;
    } else if (state == AppLifecycleState.resumed) {
      if (!_streaming) _startStream();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _fpsTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  // Map raw frame box to displayed orientation (cx,cy,w,h)
  List<double> _mapBox(double cx, double cy, double w, double h, int rotation, int rawW, int rawH) {
    switch (rotation) {
      case 90:
        return [cy, rawW - cx, h, w];
      case 270:
        return [rawH - cy, cx, h, w];
      case 180:
        return [rawW - cx, rawH - cy, w, h];
      default:
        return [cx, cy, w, h];
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('Live ${_fps.toStringAsFixed(1)} FPS  Det: $_lastDetCount'),
        actions: [
          IconButton(
            tooltip: _streaming ? 'Pause' : 'Resume',
            icon: Icon(_streaming ? Icons.pause : Icons.play_arrow),
            onPressed: () {
              if (controller == null) return;
              if (_streaming) {
                controller.stopImageStream();
                setState(() => _streaming = false);
              } else {
                _startStream();
                setState(() => _streaming = true);
              }
            },
          ),
        ],
      ),
      body: _initializing || controller == null || !controller.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(builder: (context, constraints) {
              final rotation = controller.description.sensorOrientation; // typically 90
              final previewSize = controller.value.previewSize!; // landscape
              final rawW = _frameW ?? previewSize.width.toInt();
              final rawH = _frameH ?? previewSize.height.toInt();

              // Base dims after rotation
              final rotatedW = (rotation == 90 || rotation == 270) ? rawH.toDouble() : rawW.toDouble();
              final rotatedH = (rotation == 90 || rotation == 270) ? rawW.toDouble() : rawH.toDouble();
              final rotatedAspect = rotatedW / rotatedH;

              double displayW = constraints.maxWidth;
              double displayH = displayW / rotatedAspect;
              if (displayH > constraints.maxHeight) {
                displayH = constraints.maxHeight;
                displayW = displayH * rotatedAspect;
              }

              final scaleX = displayW / rotatedW;
              final scaleY = displayH / rotatedH;

              final boxes = <Widget>[];
              for (int i = 0; i < _bboxes.length; i++) {
                final b = _bboxes[i];
                if (b.isEmpty) continue;
                final m = _mapBox(b[0], b[1], b[2], b[3], rotation, rawW, rawH);
                final sx = m[0] * scaleX;
                final sy = m[1] * scaleY;
                final sw = m[2] * scaleX;
                final sh = m[3] * scaleY;
                final cls = (i < _classes.length) ? _classes[i] : -1;
                _ensureColor(cls < 0 ? 0 : cls);
                boxes.add(Bbox(sx, sy, sw, sh, '', _scores[i], _colors[(cls < 0 ? 0 : cls) % _colors.length]));
              }

              return Center(
                child: SizedBox(
                  width: displayW,
                  height: displayH,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      // Preserve aspect ratio without stretching
                      ClipRect(
                        child: FittedBox(
                          fit: BoxFit.contain,
                          child: SizedBox(
                            width: rotatedW,
                            height: rotatedH,
                            child: CameraPreview(controller),
                          ),
                        ),
                      ),
                      ...boxes,
                      Positioned(
                        left: 8,
                        top: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '${_fps.toStringAsFixed(1)} FPS | $_lastDetCount',
                            style: const TextStyle(color: Colors.white, fontSize: 12),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }),
    );
  }
}
