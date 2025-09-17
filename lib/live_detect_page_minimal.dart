import 'dart:async';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'bbox.dart';
import 'yolo_model_helper_minimal.dart';

/// Minimal live detection page using YoloModelMinimal.
/// NOTE: This is intentionally simplified: no rotation fallback, no auto-capture,
/// no frame skipping logic beyond a basic throttle, and a naive YUV420 -> RGB conversion.
class LiveDetectPageMinimal extends StatefulWidget {
  final YoloModelMinimal model;
  const LiveDetectPageMinimal({super.key, required this.model});
  @override
  State<LiveDetectPageMinimal> createState() => _LiveDetectPageMinimalState();
}

class _LiveDetectPageMinimalState extends State<LiveDetectPageMinimal> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _initializing = true;
  bool _processing = false;
  bool _streaming = false;

  List<YoloDetection> _dets = [];
  List<Color> _colors = [];
  double _confTh = 0.4; // matches model default

  int _frames = 0;
  double _fps = 0;
  DateTime _fpsStart = DateTime.now();
  Timer? _fpsTimer;

  // Simple throttle: process every Nth frame
  static const int _processEvery = 2;
  int _frameSkip = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _initCamera();
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _computeFps());
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) { setState(()=>_initializing=false); return; }
      // Prefer back camera
      CameraDescription cam = cams.first;
      for (final c in cams) { if (c.lensDirection == CameraLensDirection.back) { cam = c; break; } }
      _controller = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      if (!mounted) return;
      setState(()=>_initializing=false);
      _start();
    } catch (e) {
      if (mounted) setState(()=>_initializing=false);
      debugPrint('Camera init error: $e');
    }
  }

  void _start() {
    if (_controller==null || _streaming) return;
    _streaming = true;
    _controller!.startImageStream(_onFrame);
  }

  void _onFrame(CameraImage frame) {
    _frames++;
    _frameSkip++;
    if (_frameSkip % _processEvery != 0) return;
    if (_processing) return;
    _processing = true;

    // Convert YUV -> RGB (naive, not optimized)
    final rgb = _yuv420ToImage(frame);
    if (rgb != null) {
      // Run inference
      final dets = widget.model.runOnImage(rgb)
          .where((d) => d.score >= _confTh)
          .toList();
      // assign colors
      for (final d in dets) _ensureColor(d.classIndex);
      if (mounted) setState(()=> _dets = dets);
    }
    _processing = false;
  }

  img.Image? _yuv420ToImage(CameraImage camImg) {
    try {
      final w = camImg.width;
      final h = camImg.height;
      final planeY = camImg.planes[0];
      final planeU = camImg.planes.length > 1 ? camImg.planes[1] : null;
      final planeV = camImg.planes.length > 2 ? camImg.planes[2] : null;
      if (planeU == null || planeV == null) return null;

      final yRowStride = planeY.bytesPerRow;
      final uvRowStride = planeU.bytesPerRow;
      final uvPixelStride = planeU.bytesPerPixel ?? 1;

      final out = img.Image(width: w, height: h);
      for (int y = 0; y < h; y++) {
        final uvRow = (y >> 1);
        for (int x = 0; x < w; x++) {
          final uvCol = (x >> 1);
          final yIndex = y * yRowStride + x;
          final uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;
          final Y = planeY.bytes[yIndex];
          final U = planeU.bytes.length > uvIndex ? planeU.bytes[uvIndex] : 128;
          final V = planeV.bytes.length > uvIndex ? planeV.bytes[uvIndex] : 128;
          final r = (Y + 1.402 * (V - 128)).clamp(0, 255).toInt();
          final g = (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)).clamp(0, 255).toInt();
          final b = (Y + 1.772 * (U - 128)).clamp(0, 255).toInt();
          out.setPixelRgba(x, y, r, g, b, 255);
        }
      }
      // Rotate for portrait if needed (most back cameras give landscape)
      final rot = _controller?.description.sensorOrientation ?? 0;
      img.Image finalImg = out;
      if (rot == 90) {
        finalImg = img.copyRotate(out, angle: 90);
      } else if (rot == 270) {
        finalImg = img.copyRotate(out, angle: -90);
      } else if (rot == 180) {
        finalImg = img.copyRotate(out, angle: 180);
      }
      return finalImg;
    } catch (e) {
      debugPrint('YUV->RGB error: $e');
      return null;
    }
  }

  void _computeFps() {
    final now = DateTime.now();
    final elapsedMs = now.difference(_fpsStart).inMilliseconds;
    if (elapsedMs > 0) {
      _fps = (_frames * 1000) / elapsedMs;
      _frames = 0;
      _fpsStart = now;
      if (mounted) setState((){});
    }
  }

  void _ensureColor(int cls) {
    if (cls >= _colors.length) {
      _colors.addAll(List<Color>.generate(cls - _colors.length + 1, (_) =>
          Color((Random().nextDouble()*0xFFFFFF).toInt()).withAlpha(255)));
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!mounted || _controller == null) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.stopImageStream();
      _streaming = false;
    } else if (state == AppLifecycleState.resumed) {
      if (!_streaming) _start();
    }
  }

  @override
  void dispose() {
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    WidgetsBinding.instance.removeObserver(this);
    _fpsTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Minimal Live YOLO'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: Icon(_streaming ? Icons.pause : Icons.play_arrow),
            onPressed: () { if (c==null) return; if (_streaming) { c.stopImageStream(); setState(()=>_streaming=false);} else { _start(); setState(()=>_streaming=true);} },
          ),
          IconButton(
            icon: const Icon(Icons.tune),
            onPressed: _showThresholdDialog,
          ),
        ],
      ),
      body: _initializing || c == null || !c.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(builder: (ctx, cons) {
              final previewSize = c.value.previewSize!; // landscape dims
              final rot = c.description.sensorOrientation; // 0/90/180/270
              final rotatedW = (rot == 90 || rot == 270) ? previewSize.height : previewSize.width;
              final rotatedH = (rot == 90 || rot == 270) ? previewSize.width : previewSize.height;
              final aspect = rotatedW / rotatedH;
              double dispW = cons.maxWidth;
              double dispH = dispW / aspect;
              if (dispH > cons.maxHeight) { dispH = cons.maxHeight; dispW = dispH * aspect; }
              final scaleX = dispW / rotatedW;
              final scaleY = dispH / rotatedH;

              List<Widget> boxes = [];
              for (final d in _dets) {
                // d coords already scaled to original frame size post-rotation logic (after rotation correction in conversion)
                final bw = d.w * scaleX;
                final bh = d.h * scaleY;
                final cx = d.cx * scaleX;
                final cy = d.cy * scaleY;
                final cls = d.classIndex;
                _ensureColor(cls);
                final color = _colors[cls % _colors.length];
                final label = (cls >=0 && cls < widget.model.labels.length) ? widget.model.labels[cls] : 'cls$cls';
                boxes.add(Bbox(cx, cy, bw, bh, label, d.score, color));
              }

              return Center(
                child: SizedBox(
                  width: dispW,
                  height: dispH,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(c),
                      ...boxes,
                      Positioned(
                        left: 8, top: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(6)),
                          child: Text('${_fps.toStringAsFixed(1)} FPS  det:${_dets.length}', style: const TextStyle(color: Colors.white, fontSize: 12)),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }),
    );
  }

  void _showThresholdDialog() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Confidence Threshold'),
        content: StatefulBuilder(
          builder: (ctx, setStateDialog) => Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('Confidence: ${_confTh.toStringAsFixed(2)}'),
              Slider(
                value: _confTh,
                min: 0.1,
                max: 0.9,
                divisions: 40,
                onChanged: (v) => setStateDialog(() => _confTh = v),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}
