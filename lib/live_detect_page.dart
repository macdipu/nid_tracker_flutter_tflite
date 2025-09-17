import 'dart:async';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  bool _autoCapture = false;

  List<int> _classes = [];
  List<List<double>> _bboxes = [];
  List<double> _scores = [];
  List<Color> _colors = [];

  double _confTh = 0.20; // Lowered threshold for better detection
  double _iouTh = 0.4;

  int? _frameW;
  int? _frameH;

  int _frames = 0;
  int _lastDetCount = 0;
  late DateTime _fpsStart;
  double _fps = 0;
  Timer? _fpsTimer;

  // Auto-capture related
  int _allClassesDetectedCount = 0;
  static const int _requiredConsecutiveDetections = 5;
  String? _targetLabel = 'nid_front_image'; // Change this to your target label
  int? _targetLabelIndex;
  bool _captureInProgress = false;

  // Performance optimization
  int _frameSkipCounter = 0;
  static const int _processEveryNFrames = 2; // Process every 2nd frame

  final GlobalKey _previewKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Lock to portrait for consistent UI
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
    ]);
    _fpsStart = DateTime.now();
    _initTargetLabelIndex();
    _initCamera();
  }

  void _initTargetLabelIndex() {
    if (widget.model.isInitialized && _targetLabel != null) {
      _targetLabelIndex = widget.model.labels.indexOf(_targetLabel!);
      if (_targetLabelIndex == -1) {
        debugPrint('⚠️ Target label "$_targetLabel" not found in model labels');
        _targetLabelIndex = null;
      } else {
        debugPrint('✅ Target label "$_targetLabel" found at index $_targetLabelIndex');
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        if (mounted) setState(() => _initializing = false);
        return;
      }

      // Try to find back camera first, fallback to first available
      CameraDescription? backCamera;
      for (final cam in cams) {
        if (cam.lensDirection == CameraLensDirection.back) {
          backCamera = cam;
          break;
        }
      }

      final cam = backCamera ?? cams.first;
      _controller = CameraController(
        cam,
        ResolutionPreset.ultraHigh, // Consider using 'high' for better detection
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();
      if (!mounted) return;

      // Set focus mode for better performance
      try {
        await _controller!.setFocusMode(FocusMode.auto);
      } catch (e) {
        debugPrint('Focus mode setting failed: $e');
      }

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

    // Skip frames for performance
    _frameSkipCounter++;
    if (_frameSkipCounter % _processEveryNFrames != 0) return;

    if (_processing || _captureInProgress) return;
    _processing = true;

    // Use lower thresholds and more aggressive processing
    widget.model
        .inferCameraImage(
      frame,
      confidenceThreshold: _confTh,
      iouThreshold: _iouTh,
      processEveryNFrames: 1, // Process every frame we send
    )
        .then((res) {
      if (!mounted || res == null) return;
      final (classes, bboxes, scores) = res;

      if (classes.isNotEmpty) {
        _lastDetCount = classes.length;
      } else if (scores.isNotEmpty) {
        debugPrint('No detections (scores: ${scores.map((s) => s.toStringAsFixed(3)).join(', ')})');
      }

      // Check if all classes are detected
      _checkAllClassesDetected(classes);

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

  void _checkAllClassesDetected(List<int> detectedClasses) {
    final totalClasses = widget.model.numClasses;
    if (totalClasses != 9) return; // Only proceed if we expect 9 classes

    // Create a set of detected classes for efficient lookup
    final detectedSet = Set<int>.from(detectedClasses);

    // Check if all classes from 0 to 8 are detected
    bool allDetected = true;
    for (int i = 0; i < 9; i++) {
      if (!detectedSet.contains(i)) {
        allDetected = false;
        break;
      }
    }

    if (allDetected) {
      _allClassesDetectedCount++;
      debugPrint('🎯 All 9 classes detected! Count: $_allClassesDetectedCount/$_requiredConsecutiveDetections');

      if (_autoCapture &&
          _allClassesDetectedCount >= _requiredConsecutiveDetections &&
          !_captureInProgress &&
          _targetLabelIndex != null) {
        _captureTargetRegion();
      }
    } else {
      _allClassesDetectedCount = 0; // Reset counter if not all detected
    }
  }

  Future<void> _captureTargetRegion() async {
    if (_captureInProgress || _targetLabelIndex == null) return;

    setState(() => _captureInProgress = true);

    try {
      // Find the target label in current detections
      int? targetDetectionIndex;
      for (int i = 0; i < _classes.length; i++) {
        if (_classes[i] == _targetLabelIndex) {
          targetDetectionIndex = i;
          break;
        }
      }

      if (targetDetectionIndex == null) {
        debugPrint('❌ Target label not found in current detections');
        setState(() => _captureInProgress = false);
        return;
      }

      // Get the bounding box for the target
      final targetBbox = _bboxes[targetDetectionIndex];

      // Capture the current camera frame
      final image = await _controller!.takePicture();

      // Here you would crop the image around the target bbox
      // For now, just show success message
      _showCaptureSuccess(targetBbox);

      debugPrint('📸 Successfully captured and cropped target region!');
      debugPrint('Target bbox: [${targetBbox.map((v) => v.toStringAsFixed(2)).join(', ')}]');

    } catch (e) {
      debugPrint('❌ Capture failed: $e');
      _showCaptureError(e.toString());
    } finally {
      setState(() => _captureInProgress = false);
      _allClassesDetectedCount = 0; // Reset for next capture
    }
  }

  void _showCaptureSuccess(List<double> bbox) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('✅ Target captured! Bbox: [${bbox.map((v) => v.toStringAsFixed(1)).join(', ')}]'),
        backgroundColor: Colors.green,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  void _showCaptureError(String error) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('❌ Capture failed: $error'),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
      ),
    );
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
    // Restore orientations
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    WidgetsBinding.instance.removeObserver(this);
    _fpsTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  // Replace _mapBox with rotation-aware center->rect transform
  List<double> _transformBoxForDisplay(List<double> bb, int rotation, int rawW, int rawH) {
    double cx = bb[0];
    double cy = bb[1];
    double w = bb[2];
    double h = bb[3];

    double tCx, tCy, tW, tH;
    switch (rotation) {
      case 90: // sensor rotated clockwise; rotate coords CCW for portrait
        tCx = cy;
        tCy = rawW - cx;
        tW = h;
        tH = w;
        break;
      case 270: // sensor rotated counter-clockwise
        tCx = rawH - cy;
        tCy = cx;
        tW = h;
        tH = w;
        break;
      case 180:
        tCx = rawW - cx;
        tCy = rawH - cy;
        tW = w;
        tH = h;
        break;
      case 0:
      default:
        tCx = cx;
        tCy = cy;
        tW = w;
        tH = h;
    }
    return [tCx, tCy, tW, tH];
  }

  Widget _buildDetectionInfo() {
    final totalClasses = widget.model.numClasses;
    final detectedClassesSet = Set<int>.from(_classes);
    final missingClasses = <int>[];

    for (int i = 0; i < totalClasses; i++) {
      if (!detectedClassesSet.contains(i)) {
        missingClasses.add(i);
      }
    }

    return Container(
      margin: const EdgeInsets.all(8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${_fps.toStringAsFixed(1)} FPS | Detected: ${_lastDetCount}/$totalClasses',
            style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
          ),
          if (_autoCapture) ...[
            const SizedBox(height: 4),
            Text(
              'Auto-capture: ${_allClassesDetectedCount >= _requiredConsecutiveDetections ? "READY" : "${_allClassesDetectedCount}/$_requiredConsecutiveDetections"}',
              style: TextStyle(
                color: _allClassesDetectedCount >= _requiredConsecutiveDetections ? Colors.green : Colors.orange,
                fontSize: 10,
              ),
            ),
          ],
          if (missingClasses.isNotEmpty && totalClasses == 9) ...[
            const SizedBox(height: 4),
            Text(
              'Missing: ${missingClasses.map((i) => widget.model.labels.length > i ? widget.model.labels[i] : 'Class$i').join(', ')}',
              style: const TextStyle(color: Colors.red, fontSize: 10),
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('Live Detection ${_captureInProgress ? "(Capturing...)" : ""}'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
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
          IconButton(
            tooltip: _autoCapture ? 'Disable Auto-Capture' : 'Enable Auto-Capture',
            icon: Icon(_autoCapture ? Icons.camera_alt : Icons.camera_alt_outlined),
            onPressed: () {
              setState(() => _autoCapture = !_autoCapture);
              if (_autoCapture && _targetLabelIndex == null) {
                _initTargetLabelIndex();
              }
            },
          ),
        ],
      ),
      body: _initializing || controller == null || !controller.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(builder: (context, constraints) {
              final rotation = controller.description.sensorOrientation; // 0,90,180,270
              final previewSize = controller.value.previewSize!;
              final rawW = _frameW ?? previewSize.width.toInt();
              final rawH = _frameH ?? previewSize.height.toInt();

              // Dimensions after rotating preview to portrait
              final rotatedW = (rotation == 90 || rotation == 270) ? rawH.toDouble() : rawW.toDouble();
              final rotatedH = (rotation == 90 || rotation == 270) ? rawW.toDouble() : rawH.toDouble();
              final aspect = rotatedW / rotatedH;

              double displayW = constraints.maxWidth;
              double displayH = displayW / aspect;
              if (displayH > constraints.maxHeight) {
                displayH = constraints.maxHeight;
                displayW = displayH * aspect;
              }

              final scaleX = displayW / rotatedW;
              final scaleY = displayH / rotatedH;

              final boxes = <Widget>[];
              for (int i = 0; i < _bboxes.length; i++) {
                final bb = _bboxes[i];
                if (bb.length < 4) continue;
                final t = _transformBoxForDisplay(bb, rotation, rawW, rawH); // [cx, cy, w, h] in raw space
                final bw = t[2] * scaleX;
                final bh = t[3] * scaleY;
                final cx = t[0] * scaleX;
                final cy = t[1] * scaleY;
                final cls = i < _classes.length ? _classes[i] : -1;
                _ensureColor(cls < 0 ? 0 : cls);
                String label = '';
                if (cls >= 0 && cls < widget.model.labels.length) label = widget.model.labels[cls];
                Color boxColor = _colors[(cls < 0 ? 0 : cls) % _colors.length];
                if (cls == _targetLabelIndex) boxColor = Colors.green;
                boxes.add(
                  Bbox(
                    cx,
                    cy,
                    bw,
                    bh,
                    label,
                    _scores[i],
                    boxColor,
                  ),
                );
              }

              // Render camera directly within the correctly sized box to avoid distortion
              return Center(
                child: SizedBox(
                  width: displayW,
                  height: displayH,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(controller),
                      ...boxes,
                      Positioned(left: 0, top: 0, child: _buildDetectionInfo()),
                      if (_captureInProgress)
                        Container(
                          color: Colors.black54,
                          child: const Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                CircularProgressIndicator(color: Colors.white),
                                SizedBox(height: 16),
                                Text('Capturing target region...', style: TextStyle(color: Colors.white, fontSize: 16)),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              );
            }),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            heroTag: "threshold",
            mini: true,
            onPressed: () => _showThresholdDialog(),
            child: const Icon(Icons.tune),
          ),
          const SizedBox(height: 8),
          FloatingActionButton(
            heroTag: "manual_capture",
            onPressed: _targetLabelIndex != null ? _captureTargetRegion : null,
            child: const Icon(Icons.camera),
          ),
        ],
      ),
    );
  }

  void _showThresholdDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Detection Thresholds'),
        content: StatefulBuilder(
          builder: (context, setDialogState) => Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('Confidence: ${_confTh.toStringAsFixed(2)}'),
              Slider(
                value: _confTh,
                min: 0.1,
                max: 0.9,
                divisions: 40,
                onChanged: (value) => setDialogState(() => _confTh = value),
              ),
              Text('IoU: ${_iouTh.toStringAsFixed(2)}'),
              Slider(
                value: _iouTh,
                min: 0.1,
                max: 0.8,
                divisions: 35,
                onChanged: (value) => setDialogState(() => _iouTh = value),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              setState(() {}); // Update main UI
              Navigator.pop(context);
            },
            child: const Text('Apply'),
          ),
        ],
      ),
    );
  }
}

