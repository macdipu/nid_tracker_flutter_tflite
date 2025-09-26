import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'bbox.dart';
import 'yolo_model_helper.dart';

///live detection page using YoloModelMinimal.
class LiveCapturePage extends StatefulWidget {
  final YoloModelHelper model;
  // Config: which label is the NID card to crop around
  final String targetLabelName;
  // Config: which labels must all be present before capture; if null => all model labels
  final List<String>? requiredLabelNames;
  // Capture only once, then pause stream and show preview
  final bool captureOnce;
  // Prefer taking a full-resolution still photo for final capture
  final bool useStillCapture;
  // JPEG quality for saved/cropped image
  final int jpegQuality;

  const LiveCapturePage({
    super.key,
    required this.model,
    this.targetLabelName = 'nid_front_image',
    this.requiredLabelNames,
    this.captureOnce = true,
    this.useStillCapture = true,
    this.jpegQuality = 95,
  });
  @override
  State<LiveCapturePage> createState() => _LiveCapturePageState();
}

class _LiveCapturePageState extends State<LiveCapturePage> with WidgetsBindingObserver {
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
  static const int _processEvery = 2;
  int _frameSkip = 0;

  // Capture state
  bool _captured = false;
  Uint8List? _capturedBytes;
  bool _capturingStill = false;

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
        ResolutionPreset.medium, // keep stream at medium for performance
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
    if (_captured && widget.captureOnce) return; // pause processing once captured
    if (_capturingStill) return; // don't process while taking still

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

      // Check capture condition: all required labels present
      final labels = widget.model.labels;
      final required = (widget.requiredLabelNames == null || widget.requiredLabelNames!.isEmpty)
          ? labels
          : widget.requiredLabelNames!;
      final detectedNames = dets.map((d) => (d.classIndex >= 0 && d.classIndex < labels.length) ? labels[d.classIndex] : 'cls${d.classIndex}')
                                .toSet();
      final allPresent = required.every(detectedNames.contains);

      if (allPresent) {
        // Find target bbox
        final targetIdx = labels.indexOf(widget.targetLabelName);
        YoloDetection? bestTarget;
        if (targetIdx >= 0) {
          for (final d in dets) {
            if (d.classIndex == targetIdx) {
              if (bestTarget == null || d.score > bestTarget.score) bestTarget = d;
            }
          }
        } else {
          // Fallback: try by name matching from detectedNames
          for (final d in dets) {
            final name = (d.classIndex >= 0 && d.classIndex < labels.length) ? labels[d.classIndex] : '';
            if (name == widget.targetLabelName) {
              if (bestTarget == null || d.score > bestTarget.score) bestTarget = d;
            }
          }
        }

        if (bestTarget != null && (!_captured || !widget.captureOnce)) {
          if (widget.useStillCapture && _controller != null) {
            // Stop stream and capture a high-res still, then crop mapped bbox
            _controller!.stopImageStream();
            _streaming = false;
            _capturingStill = true;
            if (mounted) setState(() {});
            final target = bestTarget!; // capture non-null for closures
            _captureStillAndCrop(target, rgb.width, rgb.height)
                .catchError((e) async {
                  debugPrint('Still capture failed: $e');
                  // Fallback to stream crop
                  final crop = _cropAround(rgb, target, padRatio: 0.06);
                  final bytes = Uint8List.fromList(img.encodeJpg(crop, quality: widget.jpegQuality));
                  _captured = true;
                  _capturedBytes = bytes;
                })
                .whenComplete(() {
                  _capturingStill = false;
                  if (mounted) setState(() {});
                });
            _processing = false;
            return;
          } else {
            // Crop image around bbox with small padding from stream frame
            final crop = _cropAround(rgb, bestTarget, padRatio: 0.05);
            final bytes = Uint8List.fromList(img.encodeJpg(crop, quality: widget.jpegQuality));
            _captured = true;
            _capturedBytes = bytes;
            // stop stream to freeze preview
            if (_streaming) {
              _controller?.stopImageStream();
              _streaming = false;
            }
            if (mounted) setState(() { _dets = dets; });
            _processing = false;
            return;
          }
        }
      }

      if (mounted) setState(()=> _dets = dets);
    }
    _processing = false;
  }

  Future<void> _captureStillAndCrop(YoloDetection targetOnStream, int streamW, int streamH) async {
    try {
      final c = _controller;
      if (c == null) throw StateError('Camera not initialized');
      // Take high-res photo
      final xfile = await c.takePicture();
      final bytes = await xfile.readAsBytes();
      img.Image? still = img.decodeImage(bytes);
      if (still == null) throw StateError('Failed to decode still image');
      // Respect EXIF orientation
      still = img.bakeOrientation(still);

      // Map target bbox from stream frame to still dimensions
      final scaleX = still.width / streamW;
      final scaleY = still.height / streamH;
      final mapped = YoloDetection(
        targetOnStream.classIndex,
        targetOnStream.score,
        targetOnStream.cx * scaleX,
        targetOnStream.cy * scaleY,
        targetOnStream.w * scaleX,
        targetOnStream.h * scaleY,
      );
      // Crop with a bit more padding to account for slight motion
      final crop = _cropAround(still, mapped, padRatio: 0.08);
      final out = Uint8List.fromList(img.encodeJpg(crop, quality: widget.jpegQuality));
      _captured = true;
      _capturedBytes = out;
      if (mounted) setState(() {});
    } catch (e) {
      rethrow;
    }
  }

  img.Image _cropAround(img.Image src, YoloDetection d, {double padRatio = 0.05}) {
    final left = (d.cx - d.w/2).floor();
    final top = (d.cy - d.h/2).floor();
    final right = (d.cx + d.w/2).ceil();
    final bottom = (d.cy + d.h/2).ceil();

    final pad = (max(d.w, d.h) * padRatio).round();
    int x1 = (left - pad).clamp(0, src.width - 1);
    int y1 = (top - pad).clamp(0, src.height - 1);
    int x2 = (right + pad).clamp(1, src.width);
    int y2 = (bottom + pad).clamp(1, src.height);
    final w = max(1, x2 - x1);
    final h = max(1, y2 - y1);
    return img.copyCrop(src, x: x1, y: y1, width: w, height: h);
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
      if (!_streaming && !_captured && !_capturingStill) _start();
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
    final labels = widget.model.labels;
    final required = (widget.requiredLabelNames == null || widget.requiredLabelNames!.isEmpty)
        ? labels
        : widget.requiredLabelNames!;
    final detectedNames = _dets.map((d) => (d.classIndex >= 0 && d.classIndex < labels.length) ? labels[d.classIndex] : 'cls${d.classIndex}')
        .toSet();
    final presentCount = required.where(detectedNames.contains).length;

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
                          child: Text('${_fps.toStringAsFixed(1)} FPS  det:${_dets.length}  req:${presentCount}/${required.length}', style: const TextStyle(color: Colors.white, fontSize: 12)),
                        ),
                      ),

                      if (_capturingStill) ...[
                        Container(color: Colors.black.withValues(alpha: 0.5)),
                        const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(),
                              SizedBox(height: 8),
                              Text('Capturing...', style: TextStyle(color: Colors.white)),
                            ],
                          ),
                        ),
                      ],

                      if (_captured && _capturedBytes != null) ...[
                        Container(color: Colors.black.withValues(alpha: 0.75)),
                        Align(
                          alignment: Alignment.center,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              AspectRatio(
                                aspectRatio: 3/2,
                                child: Container(
                                  constraints: const BoxConstraints(maxWidth: 360),
                                  decoration: BoxDecoration(
                                    color: Colors.black,
                                    border: Border.all(color: Colors.white24),
                                  ),
                                  child: FittedBox(
                                    fit: BoxFit.contain,
                                    child: Image.memory(_capturedBytes!),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 12),
                              Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  ElevatedButton.icon(
                                    onPressed: () {
                                      // Retake: clear capture, resume stream
                                      setState(() { _captured = false; _capturedBytes = null; });
                                      if (!_streaming) { _start(); setState(() => _streaming = true); }
                                    },
                                    icon: const Icon(Icons.refresh),
                                    label: const Text('Retake'),
                                  ),
                                  const SizedBox(width: 12),
                                  ElevatedButton.icon(
                                    onPressed: () {
                                      Navigator.of(context).pop(_capturedBytes);
                                    },
                                    icon: const Icon(Icons.check),
                                    label: const Text('Use'),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
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
